from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest_benchmark.fixture import BenchmarkFixture

from py3dtiles.tilers.point.node import Node
from py3dtiles.tilers.point.node.distance import is_point_far_enough
from py3dtiles.tilers.point.node.points_grid import Grid
from py3dtiles.typing import ExtraFieldsDescription
from py3dtiles.utils import compute_spacing, node_name_to_path

# test point
xyz = np.array([[0.25, 0.25, 0.25]], dtype=np.float32)
to_insert = np.array([[0.25, 0.25, 0.25]], dtype=np.float32)
xyz2 = np.array([[0.6, 0.6, 0.6]], dtype=np.float32)
rgb = np.array([[1, 2, 3]], dtype=np.uint8)
classification = np.array([[4]], dtype=np.uint8)
intensity = np.array([[5]], dtype=np.uint8)
sample_points = np.array(
    [[x / 30, x / 30, x / 30] for x in range(30)], dtype=np.float32
)


def test_grid_insert_simple(grid: Grid, node: Node) -> None:
    result = grid.insert(
        node.aabb[0],
        node.inv_aabb_size,
        to_insert,
        rgb,
        {"classification": classification, "intensity": intensity},
    )
    assert result is not None
    assert result[0].shape[0] == 0
    assert_array_equal(grid.cells_xyz[0], [[0.25, 0.25, 0.25]])
    assert grid.cells_rgb is not None
    assert_array_equal(grid.cells_rgb[0], [[1, 2, 3]])
    assert_array_equal(grid.cells_extra_fields["classification"][0], [4])
    assert_array_equal(grid.cells_extra_fields["intensity"][0], [5])
    result = grid.insert(
        node.aabb[0],
        node.inv_aabb_size,
        to_insert,
        rgb,
        {"classification": classification, "intensity": intensity},
    )
    assert result is not None
    assert result[0].shape[0] == 1

    # in the same cell
    result = grid.insert(
        node.aabb[0],
        node.inv_aabb_size,
        to_insert + 0.1,
        ((rgb + 1).astype(np.uint8)),
        {"classification": classification + 1, "intensity": intensity + 1},
    )
    # float arythmetic is a b***
    assert_array_almost_equal(
        grid.cells_xyz[0], np.array([[0.25, 0.25, 0.25], [0.35, 0.35, 0.35]])
    )
    assert_array_equal(grid.cells_rgb[0], [[1, 2, 3], [2, 3, 4]])
    assert_array_equal(grid.cells_extra_fields["classification"][0], [4, 5])
    assert_array_equal(grid.cells_extra_fields["intensity"][0], [5, 6])

    # in another
    result = grid.insert(
        node.aabb[0],
        node.inv_aabb_size,
        to_insert + 1,
        (rgb - 1).astype(np.uint8),
        {"classification": classification + 3, "intensity": intensity + 4},
    )
    # previous cell has not changed
    # float arythmetic is a b***
    assert_array_almost_equal(
        grid.cells_xyz[0], np.array([[0.25, 0.25, 0.25], [0.35, 0.35, 0.35]])
    )
    assert_array_equal(grid.cells_rgb[0], [[1, 2, 3], [2, 3, 4]])
    assert_array_equal(grid.cells_extra_fields["classification"][0], [4, 5])
    assert_array_equal(grid.cells_extra_fields["intensity"][0], [5, 6])
    # but another cell contain the new point
    # get the current cell
    # TODO understand why it is 21
    assert_array_almost_equal(grid.cells_xyz[21], np.array([[1.25, 1.25, 1.25]]))
    assert_array_equal(grid.cells_rgb[21], [[0, 1, 2]])
    assert_array_equal(grid.cells_extra_fields["classification"][21], [7])
    assert_array_equal(grid.cells_extra_fields["intensity"][21], [9])


def test_grid_insert_perf(grid: Grid, node: Node, benchmark: BenchmarkFixture) -> None:
    benchmark(
        grid.insert,
        node.aabb[0],
        node.inv_aabb_size,
        to_insert,
        rgb,
        {"classification": classification, "intensity": intensity},
    )


def test_grid_insert_many_points() -> None:
    half_max = 15000
    bbox = np.array([[0, 0, 0], [half_max * 4, half_max * 4, half_max * 4]])
    node = Node(
        b"noeud",
        bbox,
        compute_spacing(bbox),
        True,
        [
            ExtraFieldsDescription(name="classification", dtype=np.dtype(np.uint16)),
            ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.float32)),
        ],
    )
    for i in range(half_max):
        val1 = 2 * i
        val2 = 2 * i + 1
        xyz = np.array(
            [[val1, val1, val1], [val2, val2, val2]], dtype=np.dtype(np.float32)
        )
        rgb = np.array(
            [[val1 % 256, 0, 0], [0, val2 % 256, 0]], dtype=np.dtype(np.uint8)
        )
        classification = np.array([val1, val2], dtype=np.dtype(np.uint16))
        intensity = np.array(
            [val1 - 10000.2, val2 - 10000.2], dtype=np.dtype(np.float32)
        )
        extra_fields: dict[str, npt.NDArray[Any]] = {
            "classification": classification,
            "intensity": intensity,
        }
        node.grid.insert(
            node.aabb[0],
            node.inv_aabb_size,
            xyz,
            rgb,
            extra_fields,
        )
    # TODO make more assertions
    # TODO test balance


def test_grid_getpoints(grid: Grid, node: Node) -> None:
    grid.insert(
        node.aabb[0],
        node.inv_aabb_size,
        to_insert,
        rgb,
        {"classification": classification, "intensity": intensity},
    )
    points = grid.get_points()
    assert points is not None
    assert_array_equal(points.positions, xyz)
    assert points.colors is not None
    assert_array_equal(points.colors, [[1, 2, 3]])
    assert_array_equal(points.extra_fields["classification"], [4])
    assert_array_equal(points.extra_fields["intensity"], [5])


def test_grid_getpoints_perf(
    grid: Grid, node: Node, benchmark: BenchmarkFixture
) -> None:
    assert (
        grid.insert(
            node.aabb[0],
            node.inv_aabb_size,
            to_insert,
            rgb,
            {"classification": classification, "intensity": intensity},
        )[0].shape[0]
        == 0
    )
    benchmark(grid.get_points)


def test_grid_get_point_count_without_rgb(
    grid_position_only: Grid, node_position_only: Node
) -> None:
    grid_position_only.insert(
        node_position_only.aabb[0],
        node_position_only.inv_aabb_size,
        to_insert,
        rgb,
        {},
    )
    points = grid_position_only.get_points()
    assert points is not None
    assert points.positions.shape == (1, 3)
    assert points.colors is None
    assert points.extra_fields == {}
    grid_position_only.insert(
        node_position_only.aabb[0],
        node_position_only.inv_aabb_size,
        to_insert,
        None,
        {},
    )
    points = grid_position_only.get_points()
    assert points is not None
    assert points.positions.shape == (1, 3)


def test_grid_get_point_count_with_rgb(grid: Grid, node: Node) -> None:
    grid.insert(
        node.aabb[0],
        node.inv_aabb_size,
        to_insert,
        rgb,
        {"classification": classification, "intensity": intensity},
    )
    points = grid.get_points()
    assert points is not None
    assert points.positions.shape == (1, 3)
    assert points.extra_fields["classification"].shape == (1,)
    assert points.colors is not None
    assert points.colors.shape == (1, 3)
    grid.insert(
        node.aabb[0],
        node.inv_aabb_size,
        to_insert,
        rgb,
        {"classification": classification, "intensity": intensity},
    )
    points = grid.get_points()
    assert points is not None
    assert points.positions.shape == (1, 3)


def test_is_point_far_enough() -> None:
    points = np.array(
        [
            [1, 1, 1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
        ],
        dtype=np.dtype(np.float32),
    )
    assert not is_point_far_enough(points, xyz.ravel(), 0.25**2)
    assert is_point_far_enough(points, xyz2.ravel(), 0.25**2)


def test_is_point_far_enough_perf(benchmark: BenchmarkFixture) -> None:
    flat_xyz = xyz.ravel()
    benchmark(is_point_far_enough, sample_points, flat_xyz, 0.25**2)


def test_short_name_to_path() -> None:
    short_tile_name = b""
    path = node_name_to_path(Path("work"), short_tile_name)
    assert path == Path("work/r")


def test_long_name_to_path() -> None:
    long_tile_name = b"110542453782"
    path = node_name_to_path(Path("work"), long_tile_name)
    assert path == Path("work/11054245/r3782")


def test_long_name_to_path_with_extension() -> None:
    long_tile_name = b"110542453782"
    path = node_name_to_path(Path("work"), long_tile_name, suffix=".pnts")
    assert path == Path("work/11054245/r3782.pnts")


def test_long_name_to_path_with_short_split() -> None:
    long_tile_name = b"110542453782"
    path = node_name_to_path(Path("work"), long_tile_name, split_len=2)
    assert path == Path("work/11/05/42/45/37/r82")
