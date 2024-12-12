from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from py3dtiles.reader.las_reader import get_metadata, run
from py3dtiles.typing import ExtraFieldsDescription


class TestGetMetadata:
    def test_get_metadata_ripple(self, fixtures_dir: Path) -> None:

        path = fixtures_dir / Path("ripple.las")
        metadata = get_metadata(path)

        assert metadata["crs_in"] is None

        assert_array_almost_equal(
            metadata["aabb"],
            np.array([[-5.0, -5.0, -0.66641003], [5.0, 5.0, 1.10000002]]),
        )

        assert_array_almost_equal(
            metadata["avg_min"], np.array([-5.0, -5.0, -0.66641003])
        )
        assert metadata["extra_fields"] == [
            ExtraFieldsDescription(
                name="intensity",
                dtype=np.dtype(np.uint16),
            ),
            ExtraFieldsDescription(
                name="bit_fields",
                dtype=np.dtype(np.uint8),
            ),
            ExtraFieldsDescription(
                name="raw_classification",
                dtype=np.dtype(np.uint8),
            ),
            ExtraFieldsDescription(
                name="scan_angle_rank",
                dtype=np.dtype(np.int8),
            ),
            ExtraFieldsDescription(
                name="user_data",
                dtype=np.dtype(np.uint8),
            ),
            ExtraFieldsDescription(
                name="point_source_id",
                dtype=np.dtype(np.uint16),
            ),
            ExtraFieldsDescription(
                name="gps_time",
                dtype=np.dtype(np.float64),
            ),
        ]
        assert metadata["has_color"]
        assert metadata["point_count"] == 10201
        assert metadata["portions"] == [
            (path, (0, 10201)),
        ]

    def test_get_metadata_stacked(self, fixtures_dir: Path) -> None:

        path = fixtures_dir / Path("stacked_points.las")
        metadata = get_metadata(path)

        assert_array_almost_equal(
            metadata["aabb"], np.array([[-2.0, -2.0, -2.0], [1.0, 1.0, 1.0]])
        )

        assert_array_almost_equal(metadata["avg_min"], np.array([-2.0, -2.0, -2.0]))
        assert metadata["extra_fields"] == [
            ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.uint16)),
            ExtraFieldsDescription(name="bit_fields", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="raw_classification", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="scan_angle_rank", dtype=np.dtype(np.int8)),
            ExtraFieldsDescription(name="user_data", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="point_source_id", dtype=np.dtype(np.uint16)),
            ExtraFieldsDescription(name="gps_time", dtype=np.dtype(np.float64)),
        ]
        assert metadata["has_color"]
        assert metadata["point_count"] == 20002
        assert metadata["portions"] == [(path, (0, 20002))]

    def test_get_metadata_with_srs(self, fixtures_dir: Path) -> None:

        path = fixtures_dir / Path("with_srs_3857.las")
        metadata = get_metadata(path)

        assert_array_almost_equal(
            metadata["aabb"],
            np.array(
                [
                    [-10529552.73050467, 3668772.10846668, 87.9591828],
                    [-10528344.36710996, 3669989.96676734, 124.0741328],
                ]
            ),
        )

        assert_array_almost_equal(
            metadata["avg_min"], np.array([-10529552.730505, 3668772.108467, 87.959183])
        )
        assert metadata["extra_fields"] == [
            ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.uint16)),
            ExtraFieldsDescription(name="bit_fields", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="raw_classification", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="scan_angle_rank", dtype=np.dtype(np.int8)),
            ExtraFieldsDescription(name="user_data", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="point_source_id", dtype=np.dtype(np.uint16)),
            ExtraFieldsDescription(name="gps_time", dtype=np.dtype(np.float64)),
        ]
        assert metadata["has_color"]
        assert metadata["point_count"] == 1247
        assert metadata["portions"] == [(path, (0, 1247))]

    def test_get_metadata_without_srs(self, fixtures_dir: Path) -> None:

        path = fixtures_dir / Path("without_srs.las")
        metadata = get_metadata(path)

        assert_array_almost_equal(
            metadata["aabb"],
            np.array(
                [
                    [1.65081391e06, 8.18147914e06, 4.42600000e01],
                    [1.65081584e06, 8.18147914e06, 4.56800000e01],
                ]
            ),
        )

        assert_array_almost_equal(
            metadata["avg_min"], np.array([1.65081391e06, 8.18147914e06, 4.42600000e01])
        )
        assert metadata["extra_fields"] == [
            ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.uint16)),
            ExtraFieldsDescription(name="bit_fields", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="raw_classification", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="scan_angle_rank", dtype=np.dtype(np.int8)),
            ExtraFieldsDescription(name="user_data", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="point_source_id", dtype=np.dtype(np.uint16)),
            ExtraFieldsDescription(name="gps_time", dtype=np.dtype(np.float64)),
        ]
        assert metadata["has_color"]
        assert metadata["point_count"] == 10
        assert metadata["portions"] == [(path, (0, 10))]


class TestRun:
    def test_run_basic(self, fixtures_dir: Path) -> None:
        offset_scale = (np.array([0, 0, 0]), np.array([1, 1, 1]), None, None)
        portion = (0, 3)
        (pos, rgb, extra_fields) = next(
            run(
                str(fixtures_dir / Path("with_srs_3857.las")),
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
                    [-10529414.0, 3668815.2, 104.82],
                    [
                        -10529290.0,
                        3668810.2,
                        104.99,
                    ],
                    [
                        -10528752.0,
                        3668782.2,
                        113.56,
                    ],
                ],
                dtype=np.float32,
            ),
        )
        assert rgb is None
        assert "foo" not in extra_fields

    def test_run_file_without_rgb(self, fixtures_dir: Path) -> None:
        offset_scale = (np.array([0, 0, 0]), np.array([1, 1, 1]), None, None)
        portion = (0, 3)
        (pos, rgb, extra_fields) = next(
            run(
                str(fixtures_dir / Path("without_rgb.las")),
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
        portion = (0, 3)
        (pos, rgb, extra_fields) = next(
            run(
                str(fixtures_dir / Path("with_srs_3857.las")),
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
                    [-10529414.0, 3668815.2, 104.82],
                    [
                        -10529290.0,
                        3668810.2,
                        104.99,
                    ],
                    [
                        -10528752.0,
                        3668782.2,
                        113.56,
                    ],
                ],
                dtype=np.float32,
            ),
        )
        assert rgb is not None
        assert_array_equal(rgb, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert rgb.dtype.type == np.uint8
        assert "foo" in extra_fields
        assert_array_equal(extra_fields["foo"], [0, 0, 0])
