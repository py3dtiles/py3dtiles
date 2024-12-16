from math import ceil, log2

import numpy as np
from numba import jit  # type: ignore [attr-defined]
from numba import njit


@njit(
    "boolean(float32[:,:], float32[:], float32)", fastmath=True, nogil=True, cache=True
)
def is_point_far_enough(points, tested_point, squared_min_distance):  # pragma: no cover
    nbp = points.shape[0]
    farenough = True
    for i in range(nbp - 1, -1, -1):
        if (
            (tested_point[0] - points[i][0]) ** 2
            + (tested_point[1] - points[i][1]) ** 2
            + (tested_point[2] - points[i][2]) ** 2
        ) < squared_min_distance:
            farenough = False
            break
    return farenough


@jit(cache=True, nogil=True, nopython=True)
def xyz_to_child_index(xyz, aabb_center):  # pragma: no cover
    test = np.greater_equal(xyz - aabb_center, 0).astype(np.int8)
    return np.sum(np.left_shift(test, np.array([2, 1, 0])), axis=1)


@njit(
    "int32[:](float32[:,:], int32[:], float32[:], float32[:])",
    cache=True,
    nogil=True,
)
def xyz_to_key(xyz, cell_count, aabb_min, inv_aabb_size):  # pragma: no cover
    """
    Place all the points in xyz into a 3D xyz grid, then encode the coordinates into a single int, that can then be considered as a cell id for this specific grid.

    This grid is a subdivision of the space of cell_count size (one size for each axis), with offset aabb_min, of real inv size inv_aabb_size.

    The encoding is done by binary shifting y and z and summing x, y, z.

    For instance, for a 2-cell subdivision of a 1 by 1 by 1 cube:
    - the (1, 0, 0) point is in the cell of coordinates (1, 0, 0) and has id 1
    - (0, 1, 0) -> (0, 1, 0) -> 2 (0 + 2^1 + 0)
    - (1, 1, 1): -> (1, 1, 1) -> 7 (1 + 2 + 4)

    For a 3-cell grid:
    - (1, 1, 1) -> (2, 2, 2) -> 42 (2 + 2 * 2^2 + 2 * 2^4) (the shift is 2 because it takes 2 bits to represent 3)
    - (0.25, 0.75, 1) -> (0, 2, 2) -> 40 (0 + 2 * 2^2 + 2 * 2^4)
    - (0.5, 0.5, 0.5) -> (1, 1, 1) -> 21 (1 + 1 * 2^2 + 1 * 2^4)
    """
    shift = ceil(log2(cell_count[0]))
    a = ((cell_count * inv_aabb_size) * (xyz - aabb_min)).astype(np.int64)
    # this is a clamp
    a = np.minimum(np.maximum(a, 0), cell_count - 1)
    a[:, 1] <<= shift
    a[:, 2] <<= 2 * shift
    return np.sum(a, axis=1).astype(np.int32)
