import math
import pickle
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from py3dtiles.tilers.base_tiler import Tiler
from py3dtiles.tilers.shared_store import SharedStore
from py3dtiles.tileset import Tile
from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox

from .ifc_message_type import IfcTilerMessage, IfcWorkerMessage
from .ifc_model import FileMetadata, IfcTileInfo
from .ifc_shared_metadata import IfcSharedMetadata
from .ifc_tiler_worker import IfcTilerWorker

TILE_STOPS = [
    "IfcSite",
    "IfcBuilding",
    "ifcbuildingStorey",
    "IfcSpace",
]


@dataclass
class FilenameAndTileId:
    filename: str
    id_in_bytes: bytes


class IfcTiler(Tiler[IfcSharedMetadata, IfcTilerWorker]):
    name = "ifc"

    root_aabb: npt.NDArray[np.float64]
    node_store: SharedStore
    shared_metadata: IfcSharedMetadata
    files_to_read: list[Path]
    tiles_to_write: list[FilenameAndTileId]
    number_of_jobs: int
    verbosity: int
    cache_size: int

    def __init__(
        self,
        cache_size: int,
        verbosity: int,
        number_of_jobs: int,
    ):
        super().__init__()

        self.files_being_read: list[Path] = []
        self.tiles_to_write: list[FilenameAndTileId] = []
        self.cache_size = cache_size

        self.verbosity = verbosity
        self.number_of_jobs = number_of_jobs

        self.written_tiles_by_parent_id: dict[int, list[IfcTileInfo]] = {}
        self.files_metadata: dict[str, FileMetadata] = {}
        # that's what we're going to build folks!
        self.root_tile = Tile()

    def supports(self, file: Path) -> bool:
        # ifcopenshell advertises to support xml, json, hdf5, but the only files I manage to work with are STEP files
        return file.suffix.lower() in [".ifc"]

    def initialize(
        self,
        files: list[Path],
        working_dir: Path,
        out_folder: Path,
    ) -> None:
        self.files_to_read = files.copy()
        self.out_folder = out_folder
        self.store = SharedStore(working_dir)
        self.shared_metadata = IfcSharedMetadata(
            out_folder=self.out_folder, verbosity=self.verbosity
        )

    def get_worker(self) -> IfcTilerWorker:
        return IfcTilerWorker(self.shared_metadata)

    def get_tasks(self) -> Iterator[tuple[bytes, list[bytes]]]:
        while len(self.files_to_read) > 0:
            current_file = self.files_to_read.pop()
            self.files_being_read.append(current_file)
            yield self.send_file_to_read(current_file)
        # iterate over a copy to modify tiles_to_write in the loop
        new_tiles_to_write = []
        while len(self.tiles_to_write) > 0:
            current_info = self.tiles_to_write.pop(0)
            filename = current_info.filename
            # we cannot start working on tile until we get the metadata we need
            if filename in self.files_metadata:
                current_task = current_info.id_in_bytes
                metadata = self.files_metadata[filename]
                yield self.send_tile_to_write(current_task, metadata)
            else:
                # keep that for later
                new_tiles_to_write.append(current_info)
        self.tiles_to_write = new_tiles_to_write

    def send_tile_to_write(
        self, tile_id: bytes, metadata: FileMetadata
    ) -> tuple[bytes, list[bytes]]:
        tile = self.store.get(tile_id)
        return IfcTilerMessage.WRITE_TILE.value, [tile, pickle.dumps(metadata.offset)]

    def send_file_to_read(self, filename: Path) -> tuple[bytes, list[bytes]]:
        if self.verbosity >= 1:
            print(f"Sending {filename} to read to worker")
        return IfcTilerMessage.READ_FILE.value, [pickle.dumps({"filename": filename})]

    def add_tile_to_queue(self, tile_id: int, filename: str, tile: bytes) -> None:
        id_in_bytes = str(tile_id).encode()
        self.store.put(id_in_bytes, tile)
        self.tiles_to_write.append(
            FilenameAndTileId(filename=filename, id_in_bytes=id_in_bytes)
        )
        if self.verbosity >= 1:
            print("tiles_to_write", self.tiles_to_write)

    def add_tile_to_index(self, parent_id: int, tile: IfcTileInfo) -> None:
        if parent_id not in self.written_tiles_by_parent_id:
            self.written_tiles_by_parent_id[parent_id] = []
        self.written_tiles_by_parent_id[parent_id].append(tile)

    def create_root_tile(self, tile_metadata: IfcTileInfo) -> Tile:
        if tile_metadata.has_content:
            content_uri = (
                self.out_folder / Path(f"{tile_metadata.tile_id}.b3dm")
            ).relative_to(self.out_folder.parent)
        else:
            content_uri = None
        root_tile = Tile(content_uri=content_uri)
        if tile_metadata.box is None:
            root_tile.bounding_volume = BoundingVolumeBox()
        else:
            root_tile.bounding_volume = tile_metadata.box
        if tile_metadata.transform is not None:
            root_tile.transform = tile_metadata.transform
        root_tile.extras["id"] = tile_metadata.tile_id
        return root_tile

    def process_message(self, message_type: bytes, message: list[bytes]) -> None:
        if self.verbosity >= 1:
            print("Received message", message_type)
        if message_type == IfcWorkerMessage.TILE_PARSED.value:
            filename: str = message[0].decode()
            tile_id = cast(int, struct.unpack(">I", message[1]))
            tile_byte = message[2]
            self.add_tile_to_queue(tile_id, filename, tile_byte)
        elif message_type == IfcWorkerMessage.TILE_READY.value:
            tile_metadata: IfcTileInfo = pickle.loads(message[0])
            if self.verbosity >= 1:
                print("tile ready", tile_metadata)
            if tile_metadata.parent_id is None:
                # this is the root, let's add it right away
                self.root_tile = self.create_root_tile(tile_metadata)
            else:
                self.add_tile_to_index(tile_metadata.parent_id, tile_metadata)
        elif message_type == IfcWorkerMessage.FILE_READ.value:
            filename = message[0].decode()
            if self.verbosity >= 1:
                print("File has been read", filename)
        elif message_type == IfcWorkerMessage.METADATA_READ.value:
            filename = message[0].decode()
            metadata: FileMetadata = pickle.loads(message[1])
            self.files_metadata[filename] = metadata
        else:
            raise NotImplementedError(
                f"The command {message_type!r} is not implemented"
            )

    def create_child_tile_and_recurse(
        self, current_tile: Tile, child: IfcTileInfo
    ) -> float | None:
        content_uri = (
            Path(self.out_folder, f"{child.tile_id}.b3dm").relative_to(
                self.out_folder.parent
            )
            if child.has_content
            else None
        )
        # init child tile
        child_tile = Tile(content_uri=content_uri, bounding_volume=child.box)
        child_tile.extras["id"] = child.tile_id
        # then recurse before adding bounding volume to current_tile
        self.add_children_to_tile(child_tile)
        # now the child bounding volume and geometric error is up-to-date
        # NOTE: if we don't have bounding boxes after recursing, it means that we don't have any geometries in the subtree with this child_tile as root. We choose not to include this tile then
        if (
            isinstance(child_tile.bounding_volume, BoundingVolumeBox)
            and child_tile.bounding_volume.is_valid()
        ):
            if current_tile.bounding_volume is None:
                current_tile.bounding_volume = BoundingVolumeBox()
            current_tile.bounding_volume.add(child_tile.bounding_volume)
            if self.verbosity >= 1:
                print("elem_max_size", child.elem_max_size)
            # the geometric error is the min of all children's geometric_error
            current_tile.children.append(child_tile)
            return max(child.elem_max_size, child_tile.geometric_error)
        else:
            return 0

    def add_children_to_tile(self, current_tile: Tile) -> None:
        if self.verbosity >= 1:
            print("## add_children_to_tile", current_tile.extras)
        children: list[IfcTileInfo] | None = self.written_tiles_by_parent_id.get(
            current_tile.extras["id"]
        )
        geometric_error = 0.0
        if children is not None:
            for child in children:
                resulting_geometric_error = self.create_child_tile_and_recurse(
                    current_tile, child
                )
                if resulting_geometric_error is not None:
                    geometric_error = max(geometric_error, resulting_geometric_error)

        if self.verbosity >= 1:
            print("final geom error of ", current_tile.extras["id"], geometric_error)
        if math.isclose(geometric_error, 0.0):
            # we didn't find any geom in child tile, so they contain only metadata and offer nothing to display
            # we clear the children and set the geometric_error to 0 (leaf node)
            current_tile.geometric_error = 0
            current_tile.children = []
        else:
            current_tile.geometric_error = geometric_error

    def get_root_tile(self, use_process_pool: bool = True) -> Tile:
        self.add_children_to_tile(self.root_tile)
        return self.root_tile

    def print_summary(self) -> None:
        print("IfcTiler - summary:")
        print(f"  - files to process: {self.files_to_read}")

    def memory_control(self) -> None:
        self.store.control_memory_usage(self.cache_size, self.verbosity)
