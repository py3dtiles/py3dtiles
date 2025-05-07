from pathlib import Path

from pyproj import CRS

from py3dtiles.tilers.point.point_tiler import PointTiler


def test_get_transformer() -> None:
    point_tiler = PointTiler(
        Path("foo/"), ["dummy"], CRS("EPSG:4978"), False, False, False, None, 10, 1
    )
    point_tiler.files_info = {"crs_in": "EPSG:4978"}

    assert point_tiler.get_transformer(None) is None
    assert point_tiler.get_transformer(CRS("EPSG:4978")) is None
    assert point_tiler.get_transformer(CRS("EPSG:3857")) is not None
