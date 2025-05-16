import copy
import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from py3dtiles.tileset import BoundingVolumeBox


def test_constructor() -> None:
    bounding_volume_box = BoundingVolumeBox()
    assert bounding_volume_box is not None


def test_from_list(dummy_matrix: list[int]) -> None:
    bounding_volume_box = BoundingVolumeBox.from_list(dummy_matrix)
    box = bounding_volume_box._box
    assert_array_equal(box, np.array(dummy_matrix))  # type: ignore [arg-type]


def test_set_from_list(dummy_matrix: list[int]) -> None:
    bounding_volume_box = BoundingVolumeBox()
    bounding_volume_box.set_from_list(dummy_matrix)
    box = bounding_volume_box._box
    assert_array_equal(box, np.array(dummy_matrix))  # type: ignore [arg-type]

    with pytest.raises(ValueError):
        m2 = np.array(dummy_matrix).reshape((3, 4))
        bounding_volume_box.set_from_list(m2)


def test_set_from_invalid_list(dummy_matrix: list[int]) -> None:
    bounding_volume_box = BoundingVolumeBox()

    # Empty list
    bounding_volume_list: list[float] = []
    with pytest.raises(ValueError):
        bounding_volume_box.set_from_list(bounding_volume_list)
    assert bounding_volume_box._box is None

    # Too small list
    with pytest.raises(ValueError):
        bounding_volume_box.set_from_list(dummy_matrix[:-1])
    assert bounding_volume_box._box is None

    # Too long list
    with pytest.raises(ValueError):
        bounding_volume_box.set_from_list(dummy_matrix + [13])
    assert bounding_volume_box._box is None

    # Not only number
    with pytest.raises(ValueError):
        bounding_volume_box.set_from_list(dummy_matrix[:-1] + ["a"])
    assert bounding_volume_box._box is None

    with pytest.raises(ValueError):
        bounding_volume_box.set_from_list(dummy_matrix[:-1] + [[1]])
    assert bounding_volume_box._box is None


def test_from_points() -> None:
    bounding_volume_box = BoundingVolumeBox.from_points(
        [np.array([1, 0, 0]), np.array([2, 0, 0])]
    )
    assert bounding_volume_box._box is not None
    assert_array_equal(
        bounding_volume_box._box, np.array([1.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_set_from_points() -> None:
    bounding_volume_box = BoundingVolumeBox()
    bounding_volume_box.set_from_points([np.array([1, 0, 0]), np.array([2, 0, 0])])
    assert bounding_volume_box._box is not None
    assert_array_equal(
        bounding_volume_box._box, np.array([1.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])
    )

    bounding_volume_box = BoundingVolumeBox()
    bounding_volume_box.set_from_points(
        [
            np.array([1, 0, 0]),
            np.array([2, 0, 0]),
            np.array([1, 1, 1]),
            np.array([2, 0, -1]),
        ]
    )
    assert bounding_volume_box._box is not None
    assert_array_equal(
        bounding_volume_box._box, [1.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 1]
    )


def test_set_from_invalid_points() -> None:
    # what if I give only one point ?
    pass


def test_get_center(
    bounding_volume_box_sample: BoundingVolumeBox,
    complex_bounding_volume_box: BoundingVolumeBox,
) -> None:
    bounding_volume_box = BoundingVolumeBox()
    with pytest.raises(AttributeError):
        bounding_volume_box.get_center()

    assert_array_equal(bounding_volume_box_sample.get_center(), [0, 0, 0])

    assert_array_equal(complex_bounding_volume_box.get_center(), [1, 2, 3])


def test_translate(bounding_volume_box_sample: BoundingVolumeBox) -> None:
    bounding_volume_box = BoundingVolumeBox()
    with pytest.raises(AttributeError):
        bounding_volume_box.translate(np.array([1, 2, 3]))

    assert_array_equal(bounding_volume_box_sample.get_center(), [0, 0, 0])

    bounding_volume_box_sample.translate(np.array([1, 2, 3]))
    # Should move only the center
    # fmt: off
    expected_result = [
        1, 2, 3,
        1, 0, 0,
        0, 2, 0,
        0, 0, 3.4,
    ]
    # fmt: on
    box = bounding_volume_box_sample._box
    assert_array_equal(box, expected_result)  # type: ignore [arg-type]


def test_transform(
    dummy_matrix: list[int], bounding_volume_box_sample: BoundingVolumeBox
) -> None:
    # error case
    bounding_volume_box = BoundingVolumeBox()
    with pytest.raises(AttributeError):
        bounding_volume_box.transform(np.array(dummy_matrix))

    # Assert box hasn't change after transformation with identity matrix
    transformer = np.identity(4)
    bounding_volume_box = copy.deepcopy(bounding_volume_box_sample)
    bounding_volume_box.transform(transformer)
    assert bounding_volume_box._box is not None
    assert_array_equal(bounding_volume_box._box, dummy_matrix)

    # Assert box is translated by [10, 10, 10] on X,Y, Z axis
    transformer[:, 3] = 10
    bounding_volume_box.transform(transformer)
    # fmt: off
    expected_result = [
        10, 10, 10,
        1, 0, 0,
        0, 2, 0,
        0, 0, 3.4,
    ]
    # fmt: on
    assert_array_equal(bounding_volume_box._box, expected_result)

    # 90° rotation on z axis
    theta = math.pi / 2
    c, s = np.cos(theta), np.sin(theta)
    # fmt: off
    transformer = np.array(
        ((c, -s, 0, 0),
         (s, c, 0, 0),
         (0, 0, 1, 0),
         (0, 0, 0, 1))
    )
    # fmt: on
    bounding_volume_box = BoundingVolumeBox()
    bounding_volume_box.set_from_list([0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3])
    bounding_volume_box.transform(transformer)
    assert bounding_volume_box._box is not None
    # fmt: off
    assert_array_almost_equal(
        bounding_volume_box._box,
        [
            # same center
            0, 0, 0,
            # x,y half axis inverted
            0, 1, 0,
            -2, 0, 0,
            # same z half axis
            0, 0, 3,
        ],
    )
    # fmt: on

    # 45° rotation y axis
    theta = math.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    transformer = np.array(
        # fmt: off
        ((c, 0, -s, 0),
         (0, 1, 0, 0),
         (s, 0, c, 0),
         (0, 0, 0, 1))
        # fmt: on
    )
    bounding_volume_box = BoundingVolumeBox()
    bounding_volume_box.set_from_list([0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3])
    bounding_volume_box.transform(transformer)
    assert bounding_volume_box._box is not None
    assert_array_almost_equal(
        bounding_volume_box._box,
        # fmt: off
        [
            # same center
            0, 0, 0,
            # x axis
            c, 0, s,
            # y unchanged
            0, 2, 0,
            -3 * c, 0, 3 * s,
        ],
        # fmt: on
    )

    # -30° deg rotation on x axis
    theta = -math.pi / 3
    c, s = np.cos(theta), np.sin(theta)
    # fmt: off
    transformer = np.array(
        ((1, 0, 0, 0),
         (0, c, -s, 0),
         (0, s, c, 0),
         (0, 0, 0, 1))
    )
    # fmt: on
    bounding_volume_box = BoundingVolumeBox()
    bounding_volume_box.set_from_list([0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3])
    bounding_volume_box.transform(transformer)
    assert bounding_volume_box._box is not None
    assert_array_almost_equal(
        bounding_volume_box._box,
        # fmt: off
        [
            # same center
            0, 0, 0,
            # x axis unchanged,
            1, 0, 0,
            0, 2 * c, 2 * s,
            0, -3 * s, 3 * c,
        ],
        # fmt: on
    )


def test_get_corners(
    bounding_volume_box_sample: BoundingVolumeBox,
    complex_bounding_volume_box: BoundingVolumeBox,
) -> None:
    bounding_volume_box = BoundingVolumeBox()
    with pytest.raises(AttributeError):
        bounding_volume_box.get_corners()

    assert_array_equal(
        bounding_volume_box_sample.get_corners(),
        [  # almost a kindness test
            [-1, -2, -3.4],
            [1, -2, -3.4],
            [-1, 2, -3.4],
            [1, 2, -3.4],
            [-1, -2, 3.4],
            [1, -2, 3.4],
            [-1, 2, 3.4],
            [1, 2, 3.4],
        ],
    )

    assert_array_equal(
        complex_bounding_volume_box.get_corners(),
        [
            [-2, 3, -2],
            [0, 5, -2],
            [2, -1, -2],
            [4, 1, -2],
            [-2, 3, 8],
            [0, 5, 8],
            [2, -1, 8],
            [4, 1, 8],
        ],
    )


def test_get_canonical_as_array() -> None:
    pass


def test_to_dict(
    bounding_volume_box_sample: BoundingVolumeBox, dummy_matrix: list[int]
) -> None:
    bounding_volume_box = BoundingVolumeBox()
    with pytest.raises(AttributeError):
        bounding_volume_box.to_dict()

    assert bounding_volume_box_sample.to_dict() == {"box": dummy_matrix}


def test_is_valid(bounding_volume_box_sample: BoundingVolumeBox) -> None:
    assert bounding_volume_box_sample.is_valid()

    assert not BoundingVolumeBox().is_valid()

    bbox = BoundingVolumeBox.from_points([np.array([1, 2, 3], dtype=np.float64)])
    assert bbox.is_valid()


def test_get_half_size(
    bounding_volume_box_sample: BoundingVolumeBox,
    complex_bounding_volume_box: BoundingVolumeBox,
) -> None:
    bbox = BoundingVolumeBox()
    with pytest.raises(AttributeError):
        bbox.get_half_size()

    bbox = BoundingVolumeBox.from_points([np.array([0, 0, 0]), np.array([1, 1, 1])])
    assert_array_equal(bbox.get_half_size(), [0.5, 0.5, 0.5])

    assert all(bounding_volume_box_sample.get_half_size() == [1, 2, 3.4])
    assert_array_almost_equal(
        complex_bounding_volume_box.get_half_size(), [1.41421356, 2.82842712, 5.0]
    )


def test_union(
    bounding_volume_box_sample: BoundingVolumeBox,
    complex_bounding_volume_box: BoundingVolumeBox,
) -> None:
    newbb = BoundingVolumeBox.union(
        BoundingVolumeBox.from_points([[0, 0, 0], [1, 1, 1]]),
        BoundingVolumeBox.from_points([[0, 0, 1], [1, 1, 2]]),
    )

    assert newbb._box is not None
    assert_array_equal(
        newbb.get_corners(),
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 2],
            [1, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
        ],
    )
