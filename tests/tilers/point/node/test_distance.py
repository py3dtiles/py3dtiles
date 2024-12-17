import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_equal
from pytest import mark
from pytest_benchmark.fixture import BenchmarkFixture

from py3dtiles.tilers.point.node.distance import xyz_to_key


@mark.parametrize(
    "cell_count,expected",
    [
        (np.array([1, 1, 1], dtype=np.int32), [0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # 2 cells
        # the most significant bits are the x coordinates
        (np.array([2, 2, 2], dtype=np.int32), [0, 3, 7, 7, 1, 2, 1, 5, 6]),
        (np.array([3, 3, 3], dtype=np.int32), [0, 5, 21, 42, 2, 8, 2, 18, 40]),
    ],
)
def test_xyz_to_key(
    xyz: npt.NDArray[np.float32], cell_count: npt.NDArray[np.int32], expected: list[int]
) -> None:
    # simple case: 1 cell, size of 0, no offset
    aabb_min = np.array([0, 0, 0], dtype=np.float32)
    inv_aabb_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    assert_array_equal(
        xyz_to_key(xyz, cell_count, aabb_min, inv_aabb_size),
        expected,
    )

    # to be completed with aabb_min, inv_aabb_size variations


def test_xyz_to_key_benchmark(
    benchmark: BenchmarkFixture, xyz: npt.NDArray[np.float32]
) -> None:
    cell_count: npt.NDArray[np.int32] = np.array([1, 1, 1], dtype=np.int32)
    aabb_min = np.array([0, 0, 0], dtype=np.float32)
    inv_aabb_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    benchmark(xyz_to_key, xyz, cell_count, aabb_min, inv_aabb_size)
