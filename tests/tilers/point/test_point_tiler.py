from pyproj import CRS

from py3dtiles.tilers.point.point_tiler import PointTiler


def test_get_transformer() -> None:
    point_tiler = PointTiler(crs_in=CRS("EPSG:4978"), crs_out=None)
    point_tiler.files_info = {"crs_in": "EPSG:4978"}

    assert point_tiler.get_transformer() is None
    point_tiler.crs_out = CRS("EPSG:4978")
    assert point_tiler.get_transformer() is None
    point_tiler.crs_out = CRS("EPSG:3857")
    assert point_tiler.get_transformer() is not None
