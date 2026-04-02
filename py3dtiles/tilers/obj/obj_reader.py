from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pygltflib
import pywavefront
from pywavefront.material import Material as ObjMaterial

from py3dtiles.exceptions import InvalidInputFileError
from py3dtiles.tilers.geometry.format_reader import FormatReader
from py3dtiles.tilers.geometry.geometry_message_type import (
    GeometryWorkerMessage,
)
from py3dtiles.tilers.model import FeatureGroup, FilenameAndOffset
from py3dtiles.tileset.content.gltf_utils import GltfMesh, GltfPrimitive
from py3dtiles.utils import clamp


# inspired by https://github.com/OpenDroneMap/Obj2Tiles/blob/7f60a54b76017fde967f97c34a4fe2f6bcbbfd7f/Obj2Gltf/Converter.cs#L141
def compute_luminance(color: list[float]) -> float:
    return color[0] * 0.2125 + color[1] * 0.7154 + color[2] * 0.0721


def convert_traditional_2_metallic_roughness(mat: ObjMaterial) -> float:
    """
    convert obj material model to gltf's metallic-roughness model

    This is a port from obj2tile corresponding function
    """
    # Transform from 0-1000 range to 0-1 range. Then invert.
    roughness_factor: float = 1.0 - mat.shininess / 1000.0
    roughness_factor = clamp(roughness_factor, 0.0, 1.0)

    if mat.specular is None:
        return roughness_factor

    # Translate the blinn-phong model to the pbr metallic-roughness model
    # Roughness factor is a combination of specular intensity and shininess
    # Metallic factor is 0.0
    # Textures are not converted for now
    specular_intensity = compute_luminance(mat.specular)

    # Low specular intensity values should produce a rough material even if shininess is high.
    if specular_intensity < 0.1:
        roughness_factor *= 1.0 - specular_intensity

    return roughness_factor


def to_gltf_material(mat: ObjMaterial) -> pygltflib.Material:
    """
    This is an incomplete port of obj2tile material parsing logic
    """

    roughness_factor = convert_traditional_2_metallic_roughness(mat)

    if mat.diffuse[3] == 0:
        # there is a bug in pywavefront: when d (aka the opacity) is not
        # specified in the material, it should default to 1.0, but pywavefront
        # forgets to default it. also, if d comes *before* the diffuse color
        # declaration (which it is allowed to do), its value will be lost...
        # THIS IS A HACK, and will make d=0 material appears as fully opaque, we have lost the info definitively.
        # the real fix can only be on pywavefront side
        mat.set_alpha(1.0)

    metallic_factor = mat.specular[0]

    base_color_factor = mat.diffuse
    if base_color_factor is None:
        # this currently cannot happen because pywavefront (mistakenly) set a default color for diffuse, which is therefore never None
        # if/when it gets fixed, this code will still work
        base_color_factor = mat.ambient
    if base_color_factor is None:
        # default
        base_color_factor = [0.7, 0.7, 0.7, 1.0]

    pmr = pygltflib.PbrMetallicRoughness(
        baseColorFactor=base_color_factor,
        metallicFactor=metallic_factor,
        roughnessFactor=roughness_factor,
    )

    return pygltflib.Material(emissiveFactor=mat.emissive, pbrMetallicRoughness=pmr)


def parse_vertex_format(
    material: pywavefront.material.Material,
) -> tuple[
    npt.NDArray[np.float32] | None,
    npt.NDArray[np.float32] | None,
    npt.NDArray[np.float32] | None,
]:
    # the list of currently supported opengl formats
    # https://github.com/pywavefront/PyWavefront/blob/07d42314c9e5ed419dd6a0e75a472f7fd8d22b25/pywavefront/visualization.py#L46C1-L57C1
    # but we also ignore the "C" (color) component because it doesn't seem that wavefront supports those
    # we don't bother checking for component size, because they are all fixed
    component_offset = 0
    xyz = None
    normals = None
    uvs = None

    component_count = 0
    for part in material.vertex_format.split("_"):
        if part.startswith("V") or part.startswith("N"):
            component_count += 3
        elif part.startswith("T"):
            component_count += 2
        else:
            raise ValueError(f"Unknown format {material.vertex_format}")

    # reshape the array according to the number of components found
    vertices = np.array(material.vertices, dtype=np.float32).reshape(
        (-1, component_count)
    )
    # now parse it
    for part in material.vertex_format.split("_"):
        if part.startswith("V"):
            xyz = (vertices[:, component_offset : (component_offset + 3)]).astype(
                np.float32
            )
            component_offset += 3
        elif part.startswith("N"):
            normals = (vertices[:, component_offset : (component_offset + 3)]).astype(
                np.float32
            )
            component_offset += 3
        elif part.startswith("T"):
            uvs = (
                vertices[:, component_offset : (component_offset + 2)]
                * np.array([1, -1], dtype=np.float32)
            ).astype(np.float32)
            component_offset += 2
        else:
            raise ValueError(f"Unknown format {material.vertex_format}")

    return xyz, normals, uvs


class ObjReader(FormatReader):
    def read(
        self, filename: Path
    ) -> Iterator[tuple[GeometryWorkerMessage, Sequence[Any]]]:

        # pywavefront's logic :
        # - it puts all the vertices in scene
        # - a scene references meshes (and materials)
        # - each mesh can reference faces
        # - each mesh can reference several materials
        # - material can be shared between mesh
        #
        # So:
        # - a wavefront Scene should probably be a GltfMesh
        # - we generate a primitive for each mesh and material

        # NOTE: collect_faces parameter doesn't live up to its promise: it just
        # collects every faces into one big array at the mesh level, but
        # *still* unwrap each vertices at the material level
        # so we can't really use it :-(

        # To have something that works correctly, we'd need to do our own parser

        scene = pywavefront.Wavefront(filename, collect_faces=False)

        offset = scene.vertices[0]
        yield GeometryWorkerMessage.METADATA_READ, [
            FilenameAndOffset(filename=str(filename), offset=offset)
        ]
        members = []
        for _mesh_name, obj_mesh in scene.meshes.items():
            for material in obj_mesh.materials:
                # TODO let's not use pywavefront because it's very inefficient: it reconstructs every faces
                if obj_mesh.has_faces:
                    raise InvalidInputFileError(
                        "Currently py3dtiles doesn't support obj mesh with faces"
                    )

                try:
                    xyz, normals, uvs = parse_vertex_format(material)
                except ValueError:
                    raise InvalidInputFileError(f"Invalide file {filename}")

                if xyz is None:
                    raise InvalidInputFileError(
                        f"Cannot find vertices in file {filename}"
                    )

                texture_uri = None
                if material.texture is not None:
                    full_path = Path(material.texture.path).resolve()
                    texture_uri = str(full_path)

                gltf_material = to_gltf_material(material)
                primitive = GltfPrimitive(
                    indices=(
                        np.array(obj_mesh.faces) - 1 if obj_mesh.has_faces else None
                    ),
                    material=gltf_material,
                    texture_uri=texture_uri,
                    # note: pywavefront does not seem to supports points or lines
                    mode=pygltflib.TRIANGLES,
                )
                gltf_mesh = GltfMesh(
                    points=xyz, normals=normals, uvs=uvs, primitives=[primitive]
                )
                members.append(gltf_mesh)

        tile = FeatureGroup(
            tile_id=0, filename=filename, members=members, parent_id=None, properties={}
        )
        yield GeometryWorkerMessage.TILE_PARSED, [
            # we send tile.id separately to avoid having to unpickle data too soon on the receiver side
            str(filename),
            tile.tile_id,
            tile,
        ]
        yield GeometryWorkerMessage.FILE_READ, [str(filename).encode()]
