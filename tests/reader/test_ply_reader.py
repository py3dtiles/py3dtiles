"""Test the point cloud readers.

The example that is run in the test (`simple.ply`) comes from the [CGAL repository](https://github.com/CGAL/cgal/blob/master/Data/data/points_3/b9_training.ply). Thanks to their maintainers (for more details, please refer to CGAL, Computational Geometry Algorithms Library, https://www.cgal.org):

"""

from pathlib import Path
from typing import Any

import numpy as np
import plyfile
from numpy.testing import assert_array_equal
from pytest import raises

from py3dtiles.reader import ply_reader
from py3dtiles.typing import ExtraFieldsDescription


def test_ply_get_metadata(ply_filepath: Path) -> None:
    ply_metadata = ply_reader.get_metadata(path=ply_filepath)
    expected_point_count = 22300
    expected_aabb = (
        np.array([5.966480625e05, 2.43620015625e05, 7.350153350830078e01]),
        np.array([5.967389375e05, 2.43731984375e05, 9.718580627441406e01]),
    )
    assert list(ply_metadata.keys()) == [
        "portions",
        "aabb",
        "crs_in",
        "point_count",
        "avg_min",
        "has_color",
        "extra_fields",
    ]
    assert ply_metadata["portions"] == [(ply_filepath, (0, expected_point_count))]
    assert np.all(ply_metadata["aabb"][0] == expected_aabb[0])
    assert np.all(ply_metadata["aabb"][1] == expected_aabb[1])
    assert ply_metadata["crs_in"] is None
    assert ply_metadata["point_count"] == expected_point_count
    assert np.all(ply_metadata["avg_min"] == expected_aabb[0])
    assert ply_metadata["has_color"]
    assert ply_metadata["extra_fields"] == [
        ExtraFieldsDescription(name="label", dtype=np.dtype(np.int32))
    ]


def test_ply_get_metadata_with_extra_fields(
    ply_with_extra_fields_filepath: Path, ply_with_intensity_filepath: Path
) -> None:
    ply_metadata = ply_reader.get_metadata(
        ply_with_extra_fields_filepath,
    )
    assert ply_metadata["extra_fields"] == [
        ExtraFieldsDescription(name="classification", dtype=np.dtype(np.uint8)),
        ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.float64)),
    ]

    ply_metadata = ply_reader.get_metadata(ply_with_intensity_filepath)
    assert ply_metadata["extra_fields"] == [
        ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.uint8))
    ]


def test_ply_get_metadata_buggy(
    buggy_ply_data: dict[str, Any], buggy_ply_filepath: Path
) -> None:
    buggy_ply_data["data"].write(buggy_ply_filepath)
    with raises(KeyError, match=buggy_ply_data["msg"]):
        _ = ply_reader.get_metadata(path=buggy_ply_filepath)
    buggy_ply_filepath.unlink()


def test_create_plydata_with_renamed_property(ply_filepath: Path) -> None:
    ply_data = plyfile.PlyData.read(ply_filepath)
    modified_ply_data = ply_reader.create_plydata_with_renamed_property(
        ply_data, "label", "classification"
    )
    for prop1, prop2 in zip(
        ply_data["vertex"].properties, modified_ply_data["vertex"].properties
    ):
        assert prop1.name == prop2.name or (
            prop1.name == "label" and prop2.name == "classification"
        )
    for dtype1, dtype2 in zip(
        ply_data["vertex"].data.dtype.names,
        modified_ply_data["vertex"].data.dtype.names,
    ):
        assert dtype1 == dtype2 or (dtype1 == "label" and dtype2 == "classification")


class TestRun:
    def test_run_basic(self, fixtures_dir: Path) -> None:
        offset_scale = (np.array([0, 0, 0]), np.array([1, 1, 1]), None, None)
        portion = (0, 3, 0)
        (pos, rgb, extra_fields) = next(
            ply_reader.run(
                str(fixtures_dir / Path("simple.ply")),
                offset_scale,
                portion,
                None,
                None,
                False,
                [],
            )
        )
        np.set_printoptions(suppress=True)

        assert_array_equal(
            pos,
            np.array(
                [
                    [596732.44, 243629.12, 76.76165],
                    [596725.8, 243658.05, 76.99541],
                    [596664.4, 243696.83, 77.53805],
                ],
                dtype=np.float32,
            ),
        )
        assert rgb is None
        assert "foo" not in extra_fields

    def test_with_fake_rgb(self, fixtures_dir: Path) -> None:
        offset_scale = (np.array([0, 0, 0]), np.array([1, 1, 1]), None, None)
        portion = (0, 3, 0)
        (pos, rgb, extra_fields) = next(
            ply_reader.run(
                str(fixtures_dir / Path("simple_without_colors.ply")),
                offset_scale,
                portion,
                None,
                None,
                True,
                [],
            )
        )
        assert rgb is not None
        assert_array_equal(rgb, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert rgb.dtype.type == np.uint8

    def test_run_with_rgb_and_extrafields(self, fixtures_dir: Path) -> None:
        offset_scale = (np.array([0, 0, 0]), np.array([1, 1, 1]), None, None)
        portion = (0, 3, 0)
        (pos, rgb, extra_fields) = next(
            ply_reader.run(
                str(fixtures_dir / Path("simple.ply")),
                offset_scale,
                portion,
                None,
                None,
                True,
                [ExtraFieldsDescription(name="foo", dtype=np.dtype(np.uint8))],
            )
        )
        assert_array_equal(
            pos,
            np.array(
                [
                    [596732.44, 243629.12, 76.76165],
                    [596725.8, 243658.05, 76.99541],
                    [596664.4, 243696.83, 77.53805],
                ],
                dtype=np.float32,
            ),
        )
        assert rgb is not None
        assert_array_equal(rgb, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert rgb.dtype.type == np.uint8
        assert "foo" in extra_fields
        assert_array_equal(extra_fields["foo"], [0, 0, 0])
