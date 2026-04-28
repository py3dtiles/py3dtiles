import pickle
from collections.abc import Iterator, Sequence
from pathlib import Path

import lz4.frame as gzip
import numpy as np

from py3dtiles.constants import SpecVersion
from py3dtiles.exceptions import TilerNotFoundException
from py3dtiles.tilers.base_tiler.shared_metadata import SharedMetadata
from py3dtiles.tilers.base_tiler.tiler_worker import TilerWorker
from py3dtiles.tilers.geometry.format_reader import FormatReader
from py3dtiles.tilers.geometry.geometry_message_type import (
    GeometryTilerMessage,
    GeometryWorkerMessage,
)
from py3dtiles.tilers.model import FeatureGroup, FileMetadata, TileInfo
from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.content.b3dm import B3dm
from py3dtiles.tileset.content.gltf import Gltf
from py3dtiles.tileset.content.gltf_utils import GltfMesh

Z_UP_MATRIX_4X4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

try:
    from ..ifc.ifc_reader import IfcReader

    HAS_IFC_SUPPORT = True
except ImportError as e:
    if e.name == "ifcopenshell":
        HAS_IFC_SUPPORT = False
    else:
        raise


def get_reader(filename: Path, shared_metadata: SharedMetadata) -> FormatReader | None:
    # normally, the format support check has already been done by GeometryTiler
    if filename.suffix == ".ifc" and HAS_IFC_SUPPORT:
        return IfcReader(shared_metadata)
    else:
        return None


class GeometryTilerWorker(TilerWorker[SharedMetadata]):

    def __init__(self, shared_metadata: SharedMetadata):
        super().__init__(shared_metadata)

    def execute(
        self, command: bytes, content: list[bytes]
    ) -> Iterator[Sequence[bytes]]:
        if command == GeometryTilerMessage.READ_FILE.value:
            yield from self.execute_read_file(content)
        elif command == GeometryTilerMessage.WRITE_TILE.value:
            yield from self.write_tile_content(content)
        else:
            raise NotImplementedError(f"Unkown command {command!r}")

    def execute_read_file(self, content: list[bytes]) -> Iterator[Sequence[bytes]]:
        filename: Path = pickle.loads(content[0])
        reader = get_reader(filename, self.shared_metadata)
        if reader is None:
            raise TilerNotFoundException([filename])

        for message_type, message_content in reader.read(filename):
            binary_content = [pickle.dumps(part) for part in message_content]
            yield [message_type.value, *binary_content]

    def write_tile_content(self, content: list[bytes]) -> Iterator[Sequence[bytes]]:
        tile: FeatureGroup = pickle.loads(gzip.decompress(content[0]))
        file_metadata: FileMetadata = pickle.loads(content[1])
        offset = file_metadata.offset
        assert offset is not None  # at this point, offset *must* have been initialized
        transformer = file_metadata.transformer

        # build b3dm, write it
        meshes: list[GltfMesh] = []
        bbox = BoundingVolumeBox()
        elem_max_size = 0.0
        for mesh in tile.members:
            if transformer is not None:
                mesh.transform(transformer)
            mesh.apply_offset(offset)

            # extend the bounding box
            this_bbox = BoundingVolumeBox.from_points(mesh.points)
            bbox.add(this_bbox)
            elem_max_size = max(elem_max_size, float(np.max(this_bbox.get_half_size())))
            meshes.append(mesh)
        # no point in creating a b3dm if there is no geom in this tile
        has_content = False
        if len(meshes) > 0:
            has_content = True
            extensions = (
                "b3dm"
                if self.shared_metadata.spec_version == SpecVersion.V1_0
                else "glb"
            )
            content_path = self.shared_metadata.out_folder / Path(
                f"{tile.tile_id}.{extensions}"
            )

            tile_content: B3dm | Gltf
            if self.shared_metadata.spec_version == SpecVersion.V1_0:
                tile_content = B3dm.from_meshes(meshes, transform=Z_UP_MATRIX_4X4)

            else:
                tile_content = Gltf.from_meshes(meshes, transform=Z_UP_MATRIX_4X4)
            tile_content.save_as(content_path)

        # then create a tile of a tileset

        transform = None
        if tile.parent_id is None:
            # this is the root tile, set global transform
            transform = np.identity(4, dtype=np.float64)
            transform[0:3, 3] = offset

        tile_metadata = TileInfo(
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
        yield [GeometryWorkerMessage.TILE_READY.value, pickle.dumps(tile_metadata)]
