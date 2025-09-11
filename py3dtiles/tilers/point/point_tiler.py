import concurrent.futures
import pickle
import struct
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer

from py3dtiles.constants import CPU_COUNT, DEFAULT_CACHE_SIZE
from py3dtiles.exceptions import (
    SrsInMissingException,
    SrsInMixinException,
    TilerException,
)
from py3dtiles.tilers.base_tiler import Tiler
from py3dtiles.tilers.shared_store import SharedStore
from py3dtiles.tileset.content import read_binary_tile_content
from py3dtiles.tileset.tile import Tile
from py3dtiles.typing import ExtraFieldsDescription
from py3dtiles.utils import (
    READER_MAP,
    compute_spacing,
    make_aabb_valid,
    node_name_to_path,
)

from .matrix_manipulation import (
    make_rotation_matrix,
    make_scale_matrix,
    make_translation_matrix,
)
from .node import Node
from .pnts import MIN_POINT_SIZE, pnts_writer
from .point_message_type import PointManagerMessage, PointWorkerMessageType
from .point_shared_metadata import PointSharedMetadata
from .point_state import PointState
from .point_tiler_worker import PointTilerWorker


def is_ancestor(node_name: bytes, ancestor: bytes) -> bool:
    """
    Example, the tile 22 is ancestor of 22458
    Particular case, the tile 22 is ancestor of 22
    """
    return len(ancestor) <= len(node_name) and node_name[0 : len(ancestor)] == ancestor


def is_ancestor_in_list(
    node_name: bytes, ancestors: set[bytes] | dict[bytes, Any]
) -> bool:
    return any(
        not ancestor or is_ancestor(node_name, ancestor) for ancestor in ancestors
    )


def can_pnts_be_written(
    node_name: bytes,
    finished_node: bytes,
    input_nodes: dict[bytes, Any],
    active_nodes: set[bytes],
) -> bool:
    return (
        is_ancestor(node_name, finished_node)
        and not is_ancestor_in_list(node_name, active_nodes)
        and not is_ancestor_in_list(node_name, input_nodes)
    )


class PointTiler(Tiler[PointSharedMetadata, PointTilerWorker]):
    """
    Tiler that split pointclouds.

    This tiler is able to reproject pointclouds, and can embed arbitrary fields in the resulting 3dtiles

    :param crs_in: crs to use for files that don't have crs information in their metadata, or for all files if `force_crs_in` is used
    :param crs_out: output crs
    :param force_crs_in: whether or not to apply crs_in for all files.
    :param pyproj_always_xy: some crs defines an axis order, but some dataset still use xy order nonetheless. This boolean allows to support this case.
    :param rgb: whether to include rgb info or not
    :param color_scale: scale the color in the case of colors wrongly encoded in 8 bit in a 16-bit field (like in las/laz files).
    :param cache_size: the size in MB to use for ram cache.
    :param verbosity: verbosity level
    :param number_of_jobs: how many process this tiler is allowed to use
    :param extra_fields: the list of extra fields to include in the resulting 3dtiles
    """

    name = "points"

    files_info: dict[str, Any]
    out_folder: Path
    rgb: bool
    color_scale: float | None
    file_info: dict[str, Any]
    root_aabb: npt.NDArray[np.float64]
    root_scale: npt.NDArray[np.float32]
    root_spacing: float
    node_store: SharedStore
    state: PointState
    extra_fields_to_include: list[str]
    transformer: Transformer | None
    shared_metadata: PointSharedMetadata

    def __init__(
        self,
        crs_in: CRS | None = None,
        crs_out: CRS | None = None,
        force_crs_in: bool = False,
        pyproj_always_xy: bool = False,
        cache_size: int = DEFAULT_CACHE_SIZE,
        verbosity: int = 0,
        number_of_jobs: int = CPU_COUNT,
        rgb: bool = True,
        color_scale: float | None = None,
        extra_fields: list[str] | None = None,
    ):
        """
        Constructs a PointTiler

        """
        super().__init__(
            crs_in=crs_in,
            crs_out=crs_out,
            force_crs_in=force_crs_in,
            pyproj_always_xy=pyproj_always_xy,
            cache_size=cache_size,
            verbosity=verbosity,
            number_of_jobs=number_of_jobs,
        )

        self.rgb = rgb
        self.extra_fields_to_include = [] if extra_fields is None else extra_fields
        self.color_scale = color_scale

    def get_worker(self) -> PointTilerWorker:
        return PointTilerWorker(self.shared_metadata)

    def get_tasks(self) -> Generator[tuple[bytes, list[bytes]], None, None]:
        while len(self.state.pnts_to_writing) > 0:
            yield self.send_pnts_to_write()

        yield from self.send_points_to_process()

        while self.state.can_add_reading_jobs():
            yield self.send_file_to_read()

    def initialize(
        self, files: list[Path], working_dir: Path, out_folder: Path
    ) -> None:
        self.files = files
        self.out_folder = out_folder
        self.files_info = self.get_files_info(self.crs_in, self.force_crs_in)
        self.transformer = self.get_transformer()
        (
            self.rotation_matrix,
            self.original_aabb,
            self.avg_min,
        ) = self.get_rotation_matrix(self.crs_out, self.transformer)

        self.root_aabb, self.root_scale, self.root_spacing = self.get_root_aabb(
            self.original_aabb
        )

        self.node_store = SharedStore(working_dir)

        self.state = PointState(
            self.files_info["portions"], max(1, self.number_of_jobs // 2)
        )

        self.shared_metadata = PointSharedMetadata(
            self.transformer,
            self.root_aabb,
            self.root_spacing,
            self.root_scale,
            self.out_folder,
            self.rgb,
            self.color_scale,
            self.files_info["extra_fields"],
            self.verbosity,
        )

    def supports(self, file: Path) -> bool:
        extension = file.suffix.lower()
        return extension in READER_MAP

    def get_files_info(
        self,
        crs_in: CRS | None,
        force_crs_in: bool = False,
    ) -> dict[str, Any]:

        pointcloud_file_portions = []
        aabb = None
        total_point_count = 0
        avg_min = np.array([0.0, 0.0, 0.0])

        # read all input files headers and determine the aabb/spacing
        extra_fields_dict: dict[str, ExtraFieldsDescription] = {}
        for file in self.files:
            extension = file.suffix.lower()

            reader = READER_MAP[extension]
            file_info = reader.get_metadata(file, self.color_scale)
            extra_fields_by_name = {obj.name: obj for obj in file_info["extra_fields"]}

            if self.rgb and not file_info["has_color"]:
                print(
                    f"Warning: file ${file} does not have rgb, will default to (0, 0, 0)"
                )

            # check if we have all the asked dimensions
            for f in self.extra_fields_to_include:
                if f in extra_fields_by_name:
                    # calculate best size
                    if f in extra_fields_dict:
                        # find common dtype able to store all the different format we might have for this field
                        extra_fields_dict[f].dtype = np.promote_types(
                            extra_fields_dict[f].dtype, extra_fields_by_name[f].dtype
                        )
                    else:
                        extra_fields_dict[f] = extra_fields_by_name[f]

                else:
                    print(
                        f"Warning: the file {file} does not have the field {f}, will default to 0 or omitted if no other files have this field."
                    )

            pointcloud_file_portions += file_info["portions"]
            if aabb is None:
                aabb = file_info["aabb"]
            else:
                aabb[0] = np.minimum(aabb[0], file_info["aabb"][0])
                aabb[1] = np.maximum(aabb[1], file_info["aabb"][1])

            file_crs_in = file_info["crs_in"]
            if file_crs_in is not None:
                if crs_in is None:
                    crs_in = file_crs_in
                elif crs_in != file_crs_in and not force_crs_in:
                    raise SrsInMixinException(
                        "All input files should have the same srs in, currently there are a mix of"
                        f" {crs_in} and {file_crs_in}"
                    )
            total_point_count += file_info["point_count"]
            avg_min += file_info["avg_min"] / len(self.files)

        # The fact self.files is not empty have been checked before, so this shouldn't happen
        # but this keeps mypy happy and also serve as "defensive programming"
        if aabb is None:
            raise RuntimeError("No aabb could be computed!")
        # correct aabb, so that we don't have null sized box
        # we add 10^-5, supposing it's reasonable for most use case
        make_aabb_valid(aabb)
        return {
            "portions": pointcloud_file_portions,
            "aabb": aabb,
            "crs_in": crs_in,
            "point_count": total_point_count,
            "avg_min": avg_min,
            # note: we loop to "unwrap" the dict_values objects, which are not pickable
            "extra_fields": list(extra_fields_dict.values()),
        }

    def get_transformer(self) -> Transformer | None:
        if self.crs_out:
            if self.files_info["crs_in"] is None:
                raise SrsInMissingException(
                    "No file contains CRS in its metadata. Please specify the input crs manually."
                )
            elif self.crs_out.equals(self.files_info["crs_in"]):
                # nothing to do :-)
                transformer = None
            else:
                transformer = Transformer.from_crs(
                    self.files_info["crs_in"],
                    self.crs_out,
                    always_xy=self.pyproj_always_xy,
                )
        else:
            transformer = None

        return transformer

    def get_rotation_matrix(
        self, crs_out: CRS | None, transformer: Transformer | None
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        avg_min: npt.NDArray[np.float64] = self.files_info["avg_min"]
        aabb: npt.NDArray[np.float64] = self.files_info["aabb"]

        rotation_matrix: npt.NDArray[np.float64] = np.identity(4)
        if crs_out is not None and transformer is not None:

            bl: npt.NDArray[np.float64] = np.array(
                list(transformer.transform(aabb[0][0], aabb[0][1], aabb[0][2]))
            )
            tr: npt.NDArray[np.float64] = np.array(
                list(transformer.transform(aabb[1][0], aabb[1][1], aabb[1][2]))
            )
            br: npt.NDArray[np.float64] = np.array(
                list(transformer.transform(aabb[1][0], aabb[0][1], aabb[0][2]))
            )

            avg_min = np.array(
                list(transformer.transform(avg_min[0], avg_min[1], avg_min[2]))
            )

            x_axis = br - bl

            bl = bl - avg_min
            tr = tr - avg_min

            if crs_out.to_epsg() == 4978:
                # Transform geocentric normal => (0, 0, 1)
                # and 4978-bbox x axis => (1, 0, 0),
                # to have a bbox in local coordinates that's nicely aligned with the data
                rotation_matrix = make_rotation_matrix(avg_min, np.array([0, 0, 1]))
                rotation_matrix = np.dot(
                    make_rotation_matrix(x_axis, np.array([1, 0, 0])), rotation_matrix
                )

                rotation_matrix_part = rotation_matrix[:3, :3].T

                bl = np.dot(bl, rotation_matrix_part)
                tr = np.dot(tr, rotation_matrix_part)

            root_aabb = np.array([np.minimum(bl, tr), np.maximum(bl, tr)])
        else:
            # offset
            root_aabb = aabb - avg_min

        return rotation_matrix, root_aabb, avg_min

    def get_root_aabb(
        self, original_aabb: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float32], float]:
        base_spacing = compute_spacing(original_aabb)
        if base_spacing > 10:
            root_scale = np.array([0.01, 0.01, 0.01])
        elif base_spacing > 1:
            root_scale = np.array([0.1, 0.1, 0.1])
        else:
            root_scale = np.array([1, 1, 1])

        root_aabb = original_aabb * root_scale
        root_spacing = compute_spacing(root_aabb)
        return root_aabb, root_scale, root_spacing

    def print_summary(self) -> None:
        print("Point tiler - summary:")
        print(f"  - files to process: {self.files}")
        print("  - points to process: {}".format(self.files_info["point_count"]))
        print(f"  - offset to use: {self.avg_min}")
        print(f"  - root spacing: {self.root_spacing / self.root_scale[0]}")
        print(f"  - root aabb: {self.root_aabb}")
        print(f"  - original aabb: {self.original_aabb}")
        print(f"  - scale: {self.root_scale}")

    def send_file_to_read(self) -> tuple[bytes, list[bytes]]:
        if self.verbosity >= 1:
            print(f"Submit next portion {self.state.point_cloud_file_parts[-1]}")
        file, portion = self.state.point_cloud_file_parts.pop()
        self.state.points_in_progress += portion[1] - portion[0]

        self.state.number_of_reading_jobs += 1

        return PointManagerMessage.READ_FILE.value, [
            pickle.dumps(
                {
                    "filename": file,
                    "offset_scale": (
                        -self.avg_min,
                        self.root_scale,
                        self.rotation_matrix[:3, :3].T,
                    ),
                    "portion": portion,
                }
            ),
        ]

    def send_points_to_process(
        self,
    ) -> Generator[tuple[bytes, list[bytes]], None, None]:
        potentials = sorted(
            # a key (=task) can be in node_to_process and processing_nodes if the node isn't completely processed
            [
                (node, task)
                for node, task in self.state.node_to_process.items()  # task: [data...], point_count
                if node not in self.state.processing_nodes
            ],
            key=lambda task: -len(task[0]),
        )  # sort by node name size, the root nodes first

        while potentials:
            target_count = 100_000
            job_list = []
            count = 0
            idx = len(potentials) - 1
            while count < target_count and idx >= 0:
                name, (tasks, point_count) = potentials[idx]
                count += point_count
                job_list += [
                    name,
                    self.node_store.get(name),
                    struct.pack(">I", len(tasks)),
                ] + tasks
                del potentials[idx]

                del self.state.node_to_process[name]
                self.state.processing_nodes.add(name)

                if name in self.state.waiting_writing_nodes:
                    self.state.waiting_writing_nodes.pop(
                        self.state.waiting_writing_nodes.index(name)
                    )
                idx -= 1

            if job_list:
                yield PointManagerMessage.PROCESS_JOBS.value, job_list

    def send_pnts_to_write(self) -> tuple[bytes, list[bytes]]:
        node_name = self.state.pnts_to_writing.pop()
        data = self.node_store.get(node_name)
        if not data:
            raise ValueError(f"{node_name!r} has no data")

        self.node_store.remove(node_name)
        self.state.number_of_writing_jobs += 1

        return PointManagerMessage.WRITE_PNTS.value, [node_name, data]

    def process_message(self, message_type: bytes, message: list[bytes]) -> None:
        if message_type == PointWorkerMessageType.READ.value:
            self.state.number_of_reading_jobs -= 1

        elif message_type == PointWorkerMessageType.PROCESSED.value:
            content = pickle.loads(message[-1])
            self.state.processed_points += content["total"]
            self.state.points_in_progress -= content["total"]

            self.state.processing_nodes.remove(content["name"])

            self.dispatch_processed_nodes(content)

        elif message_type == PointWorkerMessageType.PNTS_WRITTEN.value:
            self.state.points_in_pnts += struct.unpack(">I", message[0])[0]
            self.state.number_of_writing_jobs -= 1

        elif message_type == PointWorkerMessageType.NEW_TASK.value:
            self.state.add_tasks_to_process(
                node_name=message[0],
                data=message[1],
                point_count=struct.unpack(">I", message[2])[0],
            )

        else:
            raise NotImplementedError(
                f"The command {message_type!r} is not implemented"
            )

    def dispatch_processed_nodes(self, content: dict[str, bytes]) -> None:
        if not content["name"]:
            return

        self.node_store.put(content["name"], content["data"])
        self.state.waiting_writing_nodes.append(content["name"])

        if not self.state.is_reading_finish():
            return

        # if all nodes aren't processed yet,
        # we should check if linked ancestors are processed
        if self.state.processing_nodes or self.state.node_to_process:
            finished_node = content["name"]
            if can_pnts_be_written(
                finished_node,
                finished_node,
                self.state.node_to_process,
                self.state.processing_nodes,
            ):
                self.state.waiting_writing_nodes.pop(-1)
                self.state.pnts_to_writing.append(finished_node)

                for i in range(len(self.state.waiting_writing_nodes) - 1, -1, -1):
                    candidate = self.state.waiting_writing_nodes[i]

                    if can_pnts_be_written(
                        candidate,
                        finished_node,
                        self.state.node_to_process,
                        self.state.processing_nodes,
                    ):
                        self.state.waiting_writing_nodes.pop(i)
                        self.state.pnts_to_writing.append(candidate)

        else:
            for c in self.state.waiting_writing_nodes:
                self.state.pnts_to_writing.append(c)
            self.state.waiting_writing_nodes.clear()

    def validate(self) -> None:
        if self.state.points_in_pnts != self.files_info["point_count"]:
            raise ValueError(
                "Invalid point count in the written .pnts"
                + f"(expected: {self.files_info['point_count']}, was: {self.state.points_in_pnts})"
            )

    def get_root_tile(self, use_process_pool: bool = True) -> Tile:
        # compute tile transform matrix
        transform = np.linalg.inv(self.rotation_matrix)
        transform = np.dot(transform, make_scale_matrix(1.0 / self.root_scale[0]))
        transform = np.dot(make_translation_matrix(self.avg_min), transform)

        # Create the root tile by sampling (or taking all points?) of child nodes
        root_node = Node(
            b"",
            self.root_aabb,
            self.root_spacing * 2,
            self.shared_metadata.write_rgb,
            self.shared_metadata.extra_fields_to_include,
        )
        root_node.children = []
        inv_aabb_size = (
            1.0
            / np.maximum(
                MIN_POINT_SIZE,
                self.root_aabb[1] - self.root_aabb[0],
            )
        ).astype(np.float32)
        for child_num in range(8):
            tile_path = node_name_to_path(
                self.out_folder, str(child_num).encode("ascii"), ".pnts"
            )
            if tile_path.exists():
                tile_content = read_binary_tile_content(tile_path)

                fth = tile_content.body.feature_table.header
                xyz = tile_content.body.feature_table.body.position.view(
                    np.float32
                ).reshape((fth.points_length, 3))
                if self.rgb:
                    tile_color = tile_content.body.feature_table.body.color
                    if tile_color is None:
                        raise TilerException(
                            "tile_content.body.feature_table.body.color shouldn't be None here. Seems to be a py3dtiles issue."
                        )
                    if tile_color.dtype != np.uint8:
                        raise TilerException(
                            "The data type of tile_content.body.feature_table.body.color must be np.uint8. Seems to be a py3dtiles issue."
                        )
                    rgb = tile_color.reshape((fth.points_length, 3)).astype(
                        np.uint8, copy=False
                    )  # the astype is used for typing
                else:
                    rgb = np.zeros(xyz.shape, dtype=np.uint8)

                extra_fields: dict[str, npt.NDArray[Any]] = {}
                for item in tile_content.body.batch_table.header.data.keys():
                    arr = tile_content.body.batch_table.get_binary_property(item)
                    extra_fields[item] = arr

                root_node.grid.insert(
                    self.root_aabb[0].astype(np.float32),
                    inv_aabb_size,
                    xyz.copy(),
                    rgb,
                    extra_fields,
                )

        pnts_writer.node_to_pnts(
            b"",
            root_node,
            self.out_folder,
        )

        if use_process_pool:
            pool_executor = concurrent.futures.ProcessPoolExecutor()
        else:
            pool_executor = None
        root_tile = Node.create_child_node_from_parent(
            b"",
            self.root_aabb,
            self.root_spacing,
            self.shared_metadata.write_rgb,
            self.shared_metadata.extra_fields_to_include,
        ).to_tile(self.out_folder, self.root_scale, None, 0, pool_executor)
        if pool_executor is not None:
            pool_executor.shutdown()

        if root_tile is None:
            raise RuntimeError(
                "root_tile cannot be None here. This is likely a tiler bug."
            )

        root_tile.transform = transform
        root_tile.set_refine_mode(
            "REPLACE"
        )  # The root tile is in the "REPLACE" refine mode
        # And children with the "ADD" refine mode
        # No need to set this property in their children, they will take the parent value if it is not present
        for child in root_tile.children:
            child.set_refine_mode("ADD")

        return root_tile

    def benchmark(self, benchmark_id: str, startup: float) -> None:
        print(
            "{},{},{},{}".format(
                self.benchmark,
                ",".join([f.name for f in self.files]),
                self.state.points_in_pnts,
                round(time.time() - startup, 1),
            )
        )

    def memory_control(self) -> None:
        self.node_store.control_memory_usage(self.cache_size, self.verbosity)
