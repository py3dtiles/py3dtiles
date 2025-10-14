from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Generator, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import TilerException
from py3dtiles.points import Points
from py3dtiles.tilers.point.pnts import MIN_POINT_SIZE
from py3dtiles.tilers.point.pnts.pnts_writer import points_to_pnts_file
from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.content import read_binary_tile_content
from py3dtiles.tileset.content.pnts_feature_table import SemanticPoint
from py3dtiles.tileset.tile import Tile
from py3dtiles.tileset.tileset import TileSet
from py3dtiles.typing import ExtraFieldsDescription
from py3dtiles.utils import (
    SubdivisionType,
    aabb_size_to_subdivision_type,
    node_name_to_path,
    split_aabb,
)

from .distance import xyz_to_child_index
from .points_grid import Grid

if TYPE_CHECKING:
    from typing_extensions import NotRequired

    from .node_catalog import NodeCatalog

    _T = TypeVar("_T", bound=npt.NBitBase)


def node_to_tile(
    args: tuple[Node, Path, npt.NDArray[np.float32], Node | None, int],
) -> Tile | None:
    return args[0].to_tile(args[1], args[2], args[3], args[4], None)


class DummyNodeDictType(TypedDict):
    children: NotRequired[list[bytes]]
    grid: NotRequired[Grid]
    points: NotRequired[list[Points]]


class DummyNode:
    def __init__(self, _bytes: DummyNodeDictType) -> None:
        if "children" in _bytes:
            self.children: list[bytes] | None = _bytes["children"]
            self.grid = _bytes["grid"]
        else:
            self.children = None
            self.points = _bytes["points"]

    def get_points(self) -> Points | None:
        if self.children:
            return self.grid.get_points()
        else:
            points = self.points
            if len(points) == 0:
                return None
            xyz = np.concatenate(tuple(pt.positions for pt in points))

            if points[0].colors is None:
                # assume we don't have color
                rgb: npt.NDArray[np.uint8] | None = None
            else:
                # is not none only to make mypy happy
                rgb = np.concatenate(
                    tuple(pt.colors for pt in points if pt.colors is not None)
                )

            extra_fields = {}
            for f in self.points[0].extra_fields.keys():
                extra_fields[f] = np.concatenate(
                    tuple(pt.extra_fields[f] for pt in points)
                )

            return Points(positions=xyz, colors=rgb, extra_fields=extra_fields)


class Node:
    """docstring for Node"""

    __slots__ = (
        "name",
        "aabb",
        "aabb_size",
        "inv_aabb_size",
        "aabb_center",
        "spacing",
        "include_rgb",
        "extra_fields",
        "pending_points",
        "children",
        "grid",
        "points",
        "dirty",
    )

    def __init__(
        self,
        name: bytes,
        aabb: npt.NDArray[np.float64 | np.float32],
        spacing: float,
        include_rgb: bool,
        extra_fields: list[ExtraFieldsDescription],
    ) -> None:
        super().__init__()
        self.name = name
        self.aabb = aabb.astype(
            np.float32
        )  # TODO remove astype once the whole typing is done (and once data type issues on numpy arrays are fixed).
        self.aabb_size = np.maximum(self.aabb[1] - self.aabb[0], MIN_POINT_SIZE)
        self.inv_aabb_size = 1.0 / self.aabb_size
        self.aabb_center = (self.aabb[0] + self.aabb[1]) * 0.5
        self.spacing = spacing
        self.include_rgb = include_rgb
        self.extra_fields = extra_fields
        self.pending_points: list[Points] = []
        self.children: list[bytes] | None = None
        self.grid = Grid(self.spacing, self.include_rgb, self.extra_fields)
        self.points: list[Points] = []
        self.dirty = False

    @staticmethod
    def create_child_node_from_parent(
        name: bytes,
        parent_aabb: npt.NDArray[np.floating[_T]],
        parent_spacing: float,
        include_rgb: bool,
        extra_fields: list[ExtraFieldsDescription],
    ) -> Node:
        spacing = parent_spacing * 0.5
        aabb = split_aabb(parent_aabb, int(name[-1])) if len(name) > 0 else parent_aabb
        # let's build a new Node
        return Node(name, aabb.astype(np.float64), spacing, include_rgb, extra_fields)

    def save_to_bytes(self) -> bytes:
        sub_pickle: dict[str, Any] = {}
        if self.children is not None:
            sub_pickle["children"] = self.children
            sub_pickle["grid"] = self.grid
        else:
            sub_pickle["points"] = self.points

        return pickle.dumps(sub_pickle)

    def load_from_bytes(self, byt: bytes) -> None:
        sub_pickle = pickle.loads(byt)
        if "children" in sub_pickle:
            self.children = sub_pickle["children"]
            self.grid = sub_pickle["grid"]
        else:
            self.points = sub_pickle["points"]

    def insert(
        self,
        scale: float,
        points: Points,
        make_empty_node: bool = False,
    ) -> None:
        if make_empty_node:
            self.children = []
            self.pending_points.append(points)
            return

        # fastpath
        if self.children is None:
            self.points.append(points)
            count = sum([pt.positions.shape[0] for pt in self.points])
            # stop subdividing if spacing is 1mm
            if count >= 20000 and self.spacing > 0.001 * scale:
                self._split(scale)
            self.dirty = True

            return

        # grid based insertion
        (
            remainder_xyz,
            remainder_rgb,
            remainder_extra_fields,
            needs_balance,
        ) = self.grid.insert(
            self.aabb[0],
            self.inv_aabb_size,
            points.positions,
            points.colors,
            points.extra_fields,
        )

        if needs_balance:
            self.grid.balance(self.aabb_size, self.aabb[0], self.inv_aabb_size)
            self.dirty = True

        self.dirty = self.dirty or (len(remainder_xyz) != len(points.positions))

        if len(remainder_xyz) > 0:
            self.pending_points.append(
                Points(
                    positions=remainder_xyz,
                    colors=remainder_rgb,
                    extra_fields=remainder_extra_fields,
                )
            )

    def needs_balance(self) -> bool:
        if self.children is not None:
            return self.grid.needs_balance()
        return False

    def flush_pending_points(self, catalog: NodeCatalog, scale: float) -> None:
        for name, pt in self._get_pending_points():
            catalog.get_node(name).insert(scale, pt)
        self.pending_points = []

    def dump_pending_points(self) -> list[tuple[bytes, bytes, int]]:
        result = []
        for name, pt in self._get_pending_points():
            points_dict = {
                "xyz": pt.positions,
                "rgb": pt.colors,
                "extra_fields": pt.extra_fields,
            }
            result.append((name, pickle.dumps(points_dict), len(pt.positions)))

        self.pending_points = []
        return result

    def get_pending_points_count(self) -> int:
        return sum([pt.positions.shape[0] for pt in self.pending_points])

    def _get_pending_points(
        self,
    ) -> Iterator[tuple[bytes, Points]]:
        if not self.pending_points:
            return

        pending_xyz_arr = np.concatenate([pt.positions for pt in self.pending_points])
        if self.include_rgb:
            pending_rgb_arr = np.concatenate([pt.colors for pt in self.pending_points])
        pending_extra_fields: dict[str, npt.NDArray[Any]] = {}
        for pt in self.pending_points:
            for field, arr in pt.extra_fields.items():
                if pending_extra_fields.get(field) is None:
                    pending_extra_fields[field] = arr
                else:
                    pending_extra_fields[field] = np.append(
                        pending_extra_fields[field], arr, axis=0
                    )
        t = aabb_size_to_subdivision_type(self.aabb_size)
        if t == SubdivisionType.QUADTREE:
            indices = xyz_to_child_index(
                pending_xyz_arr,
                np.array(
                    [self.aabb_center[0], self.aabb_center[1], self.aabb[1][2]],
                    dtype=np.float32,
                ),
            )
        else:
            indices = xyz_to_child_index(pending_xyz_arr, self.aabb_center)

        # unique children list
        childs = np.unique(indices)

        # make sure all children nodes exist
        for child in childs:
            name = "{}{}".format(self.name.decode("ascii"), child).encode("ascii")
            # create missing nodes, only for remembering they exist.
            # We don't want to serialize them
            # probably not needed...
            if self.children is not None and name not in self.children:
                self.children += [name]
                self.dirty = True
                # print('Added node {}'.format(name))

            mask = indices - child == 0
            xyz = pending_xyz_arr[mask]
            if len(xyz) > 0:
                extra_fields = {}
                for f, arr in pending_extra_fields.items():
                    extra_fields[f] = arr[mask]
                yield name, Points(
                    positions=xyz,
                    colors=pending_rgb_arr[mask] if self.include_rgb else None,
                    extra_fields=extra_fields,
                )

    def _split(self, scale: float) -> None:
        self.children = []
        for pt in self.points:
            self.insert(scale, pt)
        self.points = []

    def get_point_count(
        self, node_catalog: NodeCatalog, max_depth: int, depth: int = 0
    ) -> int:
        if self.children is None:
            return sum([pt.positions.shape[0] for pt in self.points])
        else:
            count = self.grid.get_point_count()
            if depth < max_depth:
                for n in self.children:
                    count += node_catalog.get_node(n).get_point_count(
                        node_catalog, max_depth, depth + 1
                    )
            return count

    def get_points(self) -> Points | None:
        if self.children is None:
            if len(self.points) == 0:
                return None
            xyz = np.concatenate(tuple(pt.positions for pt in self.points))

            if self.points[0].colors is not None:
                # the "or []" is just to make mypy happy. Normally it shouldn't be None here
                rgb: npt.NDArray[np.uint8 | np.uint16] | None = np.concatenate(
                    tuple(pt.colors or [] for pt in self.points)
                )
            else:
                rgb = None

            extra_fields = {}
            for f in self.extra_fields:
                extra_fields[f.name] = np.concatenate(
                    tuple(pt.extra_fields[f.name] for pt in self.points)
                )

            return Points(positions=xyz, colors=rgb, extra_fields=extra_fields)
        else:
            return self.grid.get_points()

    def get_child_names(self) -> Generator[bytes]:
        for number_child in range(8):
            yield f"{self.name.decode('ascii')}{number_child}".encode("ascii")

    def to_tile(
        self,
        folder: Path,
        scale: npt.NDArray[np.float32],
        parent_node: Node | None = None,
        depth: int = 0,
        pool_executor: ProcessPoolExecutor | None = None,
    ) -> Tile | None:
        # create child tileset parts
        # if their size is below of 100 points, they will be merged in this node.
        children_tileset_parts: list[Tile] = []
        parameter_to_compute: list[
            tuple[Node, Path, npt.NDArray[np.float32], Node, int]
        ] = []
        for child_name in self.get_child_names():
            child_node = Node.create_child_node_from_parent(
                child_name, self.aabb, self.spacing, self.include_rgb, self.extra_fields
            )
            child_pnts_path = node_name_to_path(folder, child_name, ".pnts")

            if child_pnts_path.exists():
                # multi thread is only allowed on nodes where there are no prune
                # a simple rule is: only is there is not a parent node
                if pool_executor and parent_node is None:
                    parameter_to_compute.append(
                        (child_node, folder, scale, self, depth + 1)
                    )
                else:
                    children_tileset_part = child_node.to_tile(
                        folder, scale, self, depth + 1
                    )
                    if (
                        children_tileset_part is not None
                    ):  # return None if the child has been merged
                        children_tileset_parts.append(children_tileset_part)

        if pool_executor and parent_node is None:
            children_tileset_parts = [
                t
                for t in pool_executor.map(node_to_tile, parameter_to_compute)
                if t is not None
            ]

        pnts_path = node_name_to_path(folder, self.name, ".pnts")
        tile_content = read_binary_tile_content(pnts_path)
        fth = tile_content.body.feature_table.header
        xyz = tile_content.body.feature_table.body.position.reshape((-1, 3))

        # check if this node should be merged in the parent.
        prune = False  # prune only if the node is a leaf

        # If this child is small enough, merge in the current tile
        if parent_node is not None and depth > 1 and fth.points_length < 100:
            parent_pnts_path = node_name_to_path(folder, parent_node.name, ".pnts")
            parent_tile = read_binary_tile_content(parent_pnts_path)
            parent_fth = parent_tile.body.feature_table.header

            parent_xyz = parent_tile.body.feature_table.body.position.reshape(
                (parent_fth.points_length, 3)
            )

            if (
                parent_fth.colors != SemanticPoint.NONE
                and parent_tile.body.feature_table.body.color is not None
            ):
                parent_rgb = parent_tile.body.feature_table.body.color.reshape((-1, 3))
            else:
                parent_rgb = None

            parent_extra_fields = {}
            for field in parent_tile.body.batch_table.header.data:
                parent_extra_fields[field] = (
                    parent_tile.body.batch_table.get_binary_property(field)
                )

            # update aabb based on real values
            parent_bounding_volume = BoundingVolumeBox.from_points(parent_xyz)

            parent_xyz = np.concatenate((parent_xyz, xyz))

            if fth.colors != SemanticPoint.NONE:
                if tile_content.body.feature_table.body.color is None:
                    raise TilerException(
                        "If the parent has color data, the children must also have color data."
                    )
                parent_rgb = np.concatenate(
                    (
                        parent_rgb,
                        tile_content.body.feature_table.body.color.reshape((-1, 3)),
                    )
                )

            for field in tile_content.body.batch_table.header.data:
                parent_extra_fields[field] = np.concatenate(
                    (
                        parent_extra_fields[field],
                        tile_content.body.batch_table.get_binary_property(field),
                    )
                )

            # update aabb
            xyz_float = xyz.view(np.float32)
            new_bounding_volume_box = BoundingVolumeBox.from_points(xyz_float)

            parent_bounding_volume.add(new_bounding_volume_box)

            parent_pnts_path.unlink()
            points_to_pnts_file(
                folder,
                parent_node.name,
                Points(
                    positions=parent_xyz,
                    colors=parent_rgb,
                    extra_fields=parent_extra_fields,
                ),
            )
            pnts_path.unlink()
            prune = True

        content_uri = None
        if not prune:
            content_uri = pnts_path.relative_to(folder)
            xyz_float = xyz.view(np.float32)

            # update aabb based on real values
            bounding_box = BoundingVolumeBox.from_points(xyz_float)

        else:
            # if it is a leaf that should be pruned
            if not children_tileset_parts:
                return None

            # recompute the aabb in function of children
            bounding_box = BoundingVolumeBox()
            for child_tileset_part in children_tileset_parts:
                if child_tileset_part.bounding_volume is not None:
                    bounding_box.add(child_tileset_part.bounding_volume)

        if bounding_box is None:
            raise TilerException("bounding_box shouldn't be None")

        tile: Tile = Tile(
            geometric_error=10 * self.spacing / scale[0], bounding_volume=bounding_box
        )
        if content_uri is not None:
            tile.content_uri = content_uri

        if children_tileset_parts:
            tile.children = children_tileset_parts
        else:
            tile.geometric_error = 0.0

        if (
            len(self.name) > 0
            and children_tileset_parts
            and len(json.dumps(tile.to_dict())) > 100000
        ):
            tile = split_tileset(tile, self.name.decode(), folder)

        return tile


def split_tileset(tile: Tile, split_name: str, folder: Path) -> Tile:
    tile.set_refine_mode("ADD")
    tileset = TileSet(geometric_error=tile.geometric_error)
    tileset.root_tile = copy.deepcopy(tile)
    tileset_name = Path(f"tileset.{split_name}.json")
    tileset.write_as_json(folder / tileset_name)
    tile.content_uri = tileset_name
    tile.children = []

    return tile
