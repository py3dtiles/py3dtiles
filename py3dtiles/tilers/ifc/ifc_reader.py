import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import ifcopenshell.geom
import ifcopenshell.util.element
import numpy as np
import pygltflib
from ifcopenshell import entity_instance

from py3dtiles.tilers.base_tiler.shared_metadata import SharedMetadata
from py3dtiles.tilers.geometry.format_reader import FormatReader
from py3dtiles.tileset.content.gltf_utils import (
    GltfMesh,
    GltfPrimitive,
)

from ..geometry.geometry_message_type import GeometryWorkerMessage
from ..model import (
    FeatureGroup,
    FilenameAndOffset,
)
from .ifc_exceptions import IfcInvalidFile


# from https://gist.github.com/stefkeB/86b6393a8d248579ac3a0fad8676d96e
def _get_children(element: entity_instance) -> list[entity_instance]:
    children = []
    # follow Spatial relation
    if element.is_a("IfcSpatialStructureElement"):
        for rel in element.ContainsElements:
            children.extend(list(rel.RelatedElements))

    # follow Aggregation Relation
    if element.is_a("IfcObjectDefinition"):
        for rel in element.IsDecomposedBy:
            children.extend(list(rel.RelatedObjects))
    return children


def convert_deg_min_sec_to_float(
    coord: tuple[int, int, int] | tuple[int, int, int, int],
) -> float:
    """
    Convert ifcsite lon or lat to float

    see https://standards.buildingsmart.org/IFC/RELEASE/IFC4_1/FINAL/HTML/schema/ifcmeasureresource/lexical/ifccompoundplaneanglemeasure.htm
    """
    float_coord = coord[0] + coord[1] / 60.0 + coord[2] / 3600.0
    # do we have the fourth component (millionth of seconds)?
    if len(coord) == 4:
        float_coord = float_coord + coord[3] / 3600.0e6
    return float_coord


def _get_elem_info(element: entity_instance) -> dict[str, Any]:
    infos = element.get_info()
    container = ifcopenshell.util.element.get_container(element)

    infos_dict = {
        "id": infos["id"],
        "class": infos["type"],
        "name": infos["Name"],
        "description": infos.get("Description", ""),
        "containerType": container.is_a() if container is not None else None,
        "containerId": container.id() if container is not None else None,
    }
    if infos["type"] == "IfcSite" and element.RefLatitude and element.RefLongitude:
        latitude = convert_deg_min_sec_to_float(element.RefLatitude)
        longitude = convert_deg_min_sec_to_float(element.RefLongitude)
        # elevation parsing
        if element.RefElevation is None:
            elevation: float = 0
        else:
            try:
                elevation = float(element.RefElevation)
            except ValueError:
                elevation = 0
        infos_dict["latitude"] = latitude
        infos_dict["longitude"] = longitude
        infos_dict["elevation"] = elevation
    return infos_dict


class IfcReader(FormatReader):

    def __init__(self, shared_metadata: SharedMetadata) -> None:
        super().__init__(shared_metadata)
        self.ifc_settings = ifcopenshell.geom.settings()  # type: ignore # mypy doesn't like small case classes apparently
        # Leaving this for reference: it does not really work (creates visual artifacts)
        # self.ifc_settings.set(ifcopenshell.geom.settings.INCLUDE_CURVES, True)
        self.ifc_settings.set(
            "use-world-coords", True
        )  # Translates and rotates the points to their world coordinates
        self.ifc_settings.set("reorient-shells", True)
        self.ifc_settings.set("apply-default-materials", False)

    def read(
        self, filename: Path
    ) -> Iterator[tuple[GeometryWorkerMessage, Sequence[Any]]]:
        # TODO current_tile_id should be a global counter
        current_tile_id = 0
        model = ifcopenshell.open(filename)
        projects = model.by_type("IfcProject")
        if len(projects) == 0:
            raise IfcInvalidFile(filename, "no IfcProject in file")

        if len(projects) > 1:
            raise IfcInvalidFile(filename, "several IfcProject found")

        # stacks of (parent_tile_id, subtree) we need to process
        # The project will constitute the root tile
        parents: list[tuple[int | None, entity_instance]] = [(None, projects[0])]
        found_offset = False
        while len(parents) > 0:
            (parent_tile_id, current_elem) = parents.pop(0)
            tile, new_parents = self.parse_tree(
                filename, current_tile_id, parent_tile_id, current_elem
            )
            if not found_offset:
                for m in tile.members:
                    # just get the first coords we find as offset, it's good enough
                    found_offset = True
                    offset = m.points[0]
                    yield GeometryWorkerMessage.METADATA_READ, [
                        FilenameAndOffset(filename=str(filename), offset=offset)
                    ]
                    break

            current_tile_id += 1
            # send the tile to main process
            yield GeometryWorkerMessage.TILE_PARSED, [
                # we send tile.id separately to avoid having to unpickle data too soon on the receiver side
                str(tile.filename),
                tile.tile_id,
                tile,
            ]
            for p in new_parents:
                parents.append((tile.tile_id, p))
        yield GeometryWorkerMessage.FILE_READ, [filename]

    def parse_elem(self, elem: entity_instance) -> GltfMesh | None:
        if hasattr(elem, "Representation") and elem.Representation is not None:
            try:
                shape = ifcopenshell.geom.create_shape(
                    self.ifc_settings,
                    elem,
                    geometry_library="hybrid-cgal-simple-opencascade",
                )  # use recommended geometry library, as described in issue #280
            except RuntimeError as e:
                if e.args[0].startswith("Failed to process shape"):
                    # this is apparently possible... but we don't really care :-)
                    return None
                else:
                    raise e
            if shape is None:
                return None

            if shape.geometry.faces:  # type: ignore
                indices = np.array(shape.geometry.faces, dtype=np.uint32).reshape(  # type: ignore
                    (-1, 3)
                )
            else:
                indices = None

            # materials
            primitives = []
            if shape.geometry.materials:  # type: ignore  # need to generate stub types for ifcopenshell
                material_ids = np.array(shape.geometry.material_ids, dtype=np.uint32)  # type: ignore
                for mat_id, m in enumerate(shape.geometry.materials):  # type: ignore
                    # TODO what to do with specular and specularity?
                    transparency = (
                        m.transparency
                        if m.transparency and not math.isnan(m.transparency)
                        else 0.0
                    )
                    base_color_factor = [
                        m.diffuse.r(),
                        m.diffuse.g(),
                        m.diffuse.b(),
                        1.0 - transparency,
                    ]
                    pbr_metallic_roughness = pygltflib.PbrMetallicRoughness(
                        # check what blender is doing
                        baseColorFactor=base_color_factor,
                        roughnessFactor=0.5,
                        metallicFactor=0.5,
                    )
                    alpha_mode = pygltflib.BLEND if transparency else pygltflib.OPAQUE
                    material = pygltflib.Material(
                        pbrMetallicRoughness=pbr_metallic_roughness,
                        alphaMode=alpha_mode,
                    )

                    if indices is not None and len(indices) > 0:
                        faces = indices.take((material_ids == mat_id).nonzero(), axis=0)

                    primitive = GltfPrimitive(indices=faces, material=material)
                    primitives.append(primitive)
            else:
                # no material, only one primitive
                primitives.append(GltfPrimitive(indices=indices, material=None))

            return GltfMesh(
                points=np.array(shape.geometry.verts, dtype=np.float64).reshape((-1, 3)),  # type: ignore
                primitives=primitives,
                properties=_get_elem_info(elem),
            )
        else:
            return None

    def parse_tree(
        self,
        filename: Path,
        tile_id: int,
        parent_tile_id: int | None,
        parent_elem: entity_instance,
    ) -> tuple[FeatureGroup, list[entity_instance]]:
        # now breadth first traversal
        # queue objects with their parent
        # each time we encounter a IfcSpatialStructureElement, this creates a new tile

        parents = []
        stack = _get_children(parent_elem)
        parent_mesh = self.parse_elem(parent_elem)
        members: list[GltfMesh] = []
        if parent_mesh is not None:
            members.append(parent_mesh)
        while len(stack) > 0:
            current = stack.pop(0)
            if current.is_a("IfcSpatialStructureElement"):
                parents.append(current)
            else:
                new_mesh = self.parse_elem(current)
                if new_mesh is not None:
                    members.append(new_mesh)
                stack.extend(_get_children(current))

        tile = FeatureGroup(
            tile_id=tile_id,
            filename=filename,
            parent_id=parent_tile_id,
            members=members,
            properties={"spatialStructure": _get_elem_info(parent_elem)},
        )
        return tile, parents
