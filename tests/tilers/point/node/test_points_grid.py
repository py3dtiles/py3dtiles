import numpy as np
from numpy.testing import assert_array_equal

from py3dtiles.tilers.point.node.points_grid import Grid


def test_init_cells() -> None:
    aabbmin = np.array([0, 0, 0], dtype=np.float32)
    inv_aabb_size = np.array([1, 1, 1], dtype=np.float32)
    xyz = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

    grid = Grid(3, False, [])

    grid.init_cells(aabbmin, inv_aabb_size, xyz, None, {})
    assert_array_equal(grid.cells_xyz[0], [[0, 0, 0]])
    # note: the point of 1,1,1 coordinates, is in the cell of coordinates (2, 2, 2)
    # for a subdivision of 3 in each axis.
    # This yields a key of 42 (2 + 2 * 2^2 + 2 * 2^4)
    # (the shift is 2 because it takes 2 bits to represent 3)
    # see the xyz_to_key method
    assert_array_equal(grid.cells_xyz[42], [[1, 1, 1]])
    # all other should be empty for now
    for idx, cell in enumerate(grid.cells_xyz):
        if idx not in (0, 42):
            assert_array_equal(cell, np.zeros((0, 3)))

    assert grid.cells_rgb is None
