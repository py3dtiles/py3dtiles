import numpy as np
from numpy.testing import assert_array_almost_equal

from py3dtiles.tilers.point.matrix_manipulation import make_rotation_matrix


def test_make_rotation_matrix() -> None:
    assert (
        make_rotation_matrix(np.array([1, 0, 0]), np.array([1, 0, 0])) == np.identity(4)
    ).all()
    assert (
        make_rotation_matrix(np.array([1, 0, 0]), np.array([23245, 0, 0]))
        == np.identity(4)
    ).all()

    assert_array_almost_equal(
        make_rotation_matrix(np.array([1, 0, 0]), np.array([0, 1, 0])),
        np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )

    # FIXME this test show that our matrix calculation is maybe completely wrong
    # assert (
    #     make_rotation_matrix(np.array([1, 0, 0]), np.array([1, 1, 0]))
    #     == np.array([[1.414213, -1.414213, 0, 0], [1.414213, 1.414213, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # ).all()
