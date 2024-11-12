from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from py3dtiles.reader.xyz_reader import get_metadata, run
from py3dtiles.typing import ExtraFieldsDescription

DATA_DIRECTORY = Path(__file__).parent.parent / "fixtures"


class TestGetMetadata:
    def test_get_metadata_simple(self) -> None:
        metadata = get_metadata(DATA_DIRECTORY / "simple.xyz")
        assert str(metadata["portions"][0][0]).endswith("tests/fixtures/simple.xyz")
        assert metadata["portions"][0][1] == (0, 3, 0)
        assert metadata["point_count"] == 3
        assert metadata["crs_in"] is None
        assert not metadata["has_color"]
        assert metadata["extra_fields"] == []

    def test_get_metadata_simple_with_irgb(self) -> None:
        metadata = get_metadata(DATA_DIRECTORY / "simple_with_irgb.xyz")
        assert str(metadata["portions"][0][0]).endswith(
            "tests/fixtures/simple_with_irgb.xyz"
        )
        assert metadata["portions"][0][1] == (0, 3, 0)
        assert metadata["has_color"]
        assert metadata["extra_fields"] == [
            ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.float32))
        ]
        assert metadata["point_count"] == 3
        assert metadata["crs_in"] is None

    def test_get_metadata_simple_with_irgb_classification(self) -> None:
        metadata = get_metadata(
            DATA_DIRECTORY / "simple_with_irgb_and_classification.csv"
        )
        assert str(metadata["portions"][0][0]).endswith(
            "tests/fixtures/simple_with_irgb_and_classification.csv"
        )
        assert metadata["portions"][0][1] == (0, 3, 20)
        assert metadata["has_color"]
        assert metadata["extra_fields"] == [
            ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.float32)),
            ExtraFieldsDescription(name="classification", dtype=np.dtype(np.uint32)),
        ]
        assert metadata["point_count"] == 3
        assert metadata["crs_in"] is None


class TestRun:
    def test_run_basic(self, fixtures_dir: Path) -> None:
        offset_scale = (np.array([0, 0, 0]), np.array([1, 1, 1]), None, None)
        portion = (0, 3, 0)
        (pos, rgb, extra_fields) = next(
            run(
                fixtures_dir / Path("simple.xyz"),
                offset_scale,
                portion,
                None,
                None,
                False,
                [],
            )
        )
        assert_array_equal(
            pos,
            np.array(
                [
                    [281345.12, 369123.47, 0.0],
                    [281345.56, 369123.88, 0.0],
                    [281345.9, 369123.25, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        assert rgb is None
        assert "foo" not in extra_fields

    def test_run_with_rgb_and_extrafields(self, fixtures_dir: Path) -> None:
        offset_scale = (np.array([0, 0, 0]), np.array([1, 1, 1]), None, None)
        portion = (0, 3, 0)
        (pos, rgb, extra_fields) = next(
            run(
                fixtures_dir / Path("simple.xyz"),
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
                    [281345.12, 369123.47, 0.0],
                    [281345.56, 369123.88, 0.0],
                    [281345.9, 369123.25, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        assert rgb is not None
        assert_array_equal(rgb, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert "foo" in extra_fields
        assert_array_equal(extra_fields["foo"], [0, 0, 0])
