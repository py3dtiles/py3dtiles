import pickle
import struct
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import ifcopenshell.geom
import ifcopenshell.util.element
import lz4.frame as gzip
import numpy as np
from ifcopenshell import entity_instance

from py3dtiles.tilers.base_tiler.tiler_worker import TilerWorker
from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.content.b3dm import B3dm
from py3dtiles.tileset.content.gltf_utils import GltfMesh, GltfPrimitive

from .ifc_exceptions import IfcInvalidFile
from .ifc_message_type import IfcTilerMessage, IfcWorkerMessage
from .ifc_model import (
    Color,
    Feature,
    FileMetadata,
    FilenameAndOffset,
    Geometry,
    IfcMaterial,
    IfcMesh,
    IfcTile,
    IfcTileInfo,
)
from .ifc_shared_metadata import IfcSharedMetadata

Z_UP_MATRIX_4X4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


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


def _get_elem_info(
    element: entity_instance, container: entity_instance | None
) -> dict[str, Any]:
    infos = element.get_info()

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


class IfcTilerWorker(TilerWorker[IfcSharedMetadata]):

    def __init__(self, shared_metadata: IfcSharedMetadata):
        super().__init__(shared_metadata)

    def initialize(self) -> None:
        self.ifc_settings = ifcopenshell.geom.settings()  # type: ignore # mypy doesn't like small case classes apparently
        # Leaving this for reference: it does not really work (creates visual artifacts)
        # self.ifc_settings.set(ifcopenshell.geom.settings.INCLUDE_CURVES, True)
        self.ifc_settings.set(
            "use-world-coords", True
        )  # Translates and rotates the points to their world coordinates
        self.ifc_settings.set("reorient-shells", True)
        self.ifc_settings.set("apply-default-materials", False)

    def execute(
        self, command: bytes, content: list[bytes]
    ) -> Iterator[Sequence[bytes]]:
        if command == IfcTilerMessage.READ_FILE.value:
            yield from self.execute_read_file(content)
        elif command == IfcTilerMessage.WRITE_TILE.value:
            yield from self.write_tile_content(content)
        else:
            raise NotImplementedError(f"Unkown command {command!r}")

    def execute_read_file(self, content: list[bytes]) -> Iterator[Sequence[bytes]]:
        current_tile_id = 0
        filename: Path = pickle.loads(content[0])
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
                    if m.mesh:
                        # just get the first coords we find as offset, it's good enough
                        found_offset = True
                        offset = m.mesh.geom.verts[0:3]
                        yield [
                            IfcWorkerMessage.METADATA_READ.value,
                            pickle.dumps(
                                FilenameAndOffset(filename=str(filename), offset=offset)
                            ),
                        ]
                        break

            current_tile_id += 1
            # send the tile to main process
            yield [
                IfcWorkerMessage.TILE_PARSED.value,
                # we send tile.id separately to avoid having to unpickle data too soon on the receiver side
                str(tile.filename).encode(),
                struct.pack(">I", tile.tile_id),
                pickle.dumps(tile),
            ]
            for p in new_parents:
                parents.append((tile.tile_id, p))
        yield [IfcWorkerMessage.FILE_READ.value, str(filename).encode()]

    def write_tile_content(self, content: list[bytes]) -> Iterator[Sequence[bytes]]:
        tile: IfcTile = pickle.loads(gzip.decompress(content[0]))
        file_metadata: FileMetadata = pickle.loads(content[1])
        offset = file_metadata.offset
        assert offset is not None  # at this point, offset *must* have been initialized
        transformer = file_metadata.transformer

        # build b3dm, write it
        meshes: list[GltfMesh] = []
        bbox = BoundingVolumeBox()
        elem_max_size = 0.0
        for f in tile.members:
            if f.mesh is not None:
                pts = np.array(f.mesh.geom.verts, dtype=np.float64).reshape((-1, 3))
                if transformer is not None:
                    xx, yy, zz = transformer.transform(pts[:, 0], pts[:, 1], pts[:, 2])
                    pts = np.vstack((xx, yy, zz)).transpose()
                points = (pts - offset).astype(np.float32)

                # extend the bounding box
                this_bbox = BoundingVolumeBox.from_points(points)
                bbox.add(this_bbox)
                elem_max_size = max(
                    elem_max_size, float(np.max(this_bbox.get_half_size()))
                )
                if f.mesh.geom.faces:
                    triangles = np.array(f.mesh.geom.faces, dtype=np.uint32).reshape(
                        (-1, 3)
                    )

                primitives = []
                if f.mesh.materials:
                    material_ids = np.array(f.mesh.material_ids, dtype=np.uint32)
                    for mat_id in range(len(f.mesh.materials)):
                        material = f.mesh.materials[mat_id].to_pygltflib_material()
                        if f.mesh.geom.faces:
                            tri = triangles.take(
                                (material_ids == mat_id).nonzero(), axis=0
                            )
                        primitive = GltfPrimitive(triangles=tri, material=material)
                        primitives.append(primitive)
                else:
                    # no material, only one primitive
                    primitives.append(GltfPrimitive(triangles=triangles))
                    material = None

                # let' s split the vertices and triangles by material_ids
                meshes.append(
                    GltfMesh(points, primitives=primitives, properties=f.properties)
                )
        # no point in creating a b3dm if there is no geom in this tile
        has_content = False
        b3dm_path = None
        if len(meshes) > 0:
            has_content = True
            b3dm_path = self.shared_metadata.out_folder / Path(f"{tile.tile_id}.b3dm")

            tile_content = B3dm.from_meshes(meshes, transform=Z_UP_MATRIX_4X4)
            tile_content.sync()
            tile_content.save_as(b3dm_path)

        # then create a tile of a tileset

        transform = None
        if tile.parent_id is None:
            # this is the root tile, set global transform
            transform = np.identity(4, dtype=np.float64)
            transform[0:3, 3] = offset

        tile_metadata = IfcTileInfo(
            tile_id=tile.tile_id,
            parent_id=tile.parent_id,
            box=bbox,
            transform=transform,
            has_content=has_content,
            elem_max_size=elem_max_size,
            properties=tile.properties,
        )

        if self.shared_metadata.verbosity >= 1:
            print("sending tile ready with metadata", tile_metadata)
        yield [IfcWorkerMessage.TILE_READY.value, pickle.dumps(tile_metadata)]

    def parse_elem(self, elem: entity_instance) -> IfcMesh | None:
        if hasattr(elem, "Representation") and elem.Representation is not None:
            try:
                shape = ifcopenshell.geom.create_shape(self.ifc_settings, elem)
            except RuntimeError as e:
                if e.args[0].startswith("Failed to process shape"):
                    # this is apparently possible... but we don't really care :-)
                    return None
                else:
                    raise e
            if shape is None:
                return None

            # materials
            materials = []
            for m in shape.geometry.materials:  # type: ignore  # need to generate stub types for ifcopenshell

                material = IfcMaterial(
                    diffuse=Color(r=m.diffuse.r(), g=m.diffuse.g(), b=m.diffuse.b()),
                    specular=Color(
                        r=m.specular.r(), g=m.specular.g(), b=m.specular.b()
                    ),
                    specularity=m.specularity,
                    transparency=m.transparency,
                )
                materials.append(material)
            return IfcMesh(
                geom=Geometry(
                    verts=shape.geometry.verts,  # type: ignore  # ifcopenshell needs to fix their type declarations
                    faces=shape.geometry.faces,  # type: ignore
                ),
                materials=materials,
                material_ids=shape.geometry.material_ids,  # type: ignore
            )
        else:
            return None

    def parse_tree(
        self,
        filename: Path,
        tile_id: int,
        parent_tile_id: int | None,
        parent_elem: entity_instance,
    ) -> tuple[IfcTile, list[entity_instance]]:
        # now breadth first traversal
        # queue objects with their parent
        # each time we encounter a IfcSpatialStructureElement, this creates a new tile

        parents = []
        stack = _get_children(parent_elem)
        parent_mesh = self.parse_elem(parent_elem)
        parent_feature = Feature(
            mesh=parent_mesh,
            properties=_get_elem_info(
                parent_elem, ifcopenshell.util.element.get_container(parent_elem)
            ),
        )
        members: list[Feature] = [parent_feature]
        while len(stack) > 0:
            current = stack.pop(0)
            if current.is_a("IfcSpatialStructureElement"):
                parents.append(current)
            else:
                new_mesh = self.parse_elem(current)
                new_feat = Feature(
                    mesh=new_mesh, properties=_get_elem_info(current, parent_elem)
                )
                members.append(new_feat)
                stack.extend(_get_children(current))

        tile = IfcTile(
            tile_id=tile_id,
            filename=filename,
            parent_id=parent_tile_id,
            members=members,
            properties={"spatialStructure": parent_feature.properties},
        )
        return tile, parents
