import copy
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import plyfile
from plyfile import PlyData, PlyElement
from pyproj import CRS
from pytest import fixture

from py3dtiles.convert import convert
from py3dtiles.tilers.b3dm.wkb_utils import PolygonType
from py3dtiles.tilers.point.node import Grid, Node
from py3dtiles.tileset import BoundingVolumeBox, Tile, TileSet
from py3dtiles.tileset.extension.batch_table_hierarchy_extension import (
    BatchTableHierarchy,
)
from py3dtiles.typing import ExtraFieldsDescription
from py3dtiles.utils import compute_spacing

from .fixtures.mock_extension import MockExtension


@fixture()
def tmp_dir() -> Iterator[Path]:
    tmp_dir = Path("tmp/")
    tmp_dir.mkdir()
    yield tmp_dir
    if tmp_dir.exists():
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            tmp_dir.unlink()


# It can be tempting to replace that with python API about temporary file, but please test on gitlab ci:
# last time (2025-10) I checked, we couldn't write on the temp folder of the gitlab.com windows runner...
@fixture(scope="session")
def tmp_fixture_dir() -> Iterator[Path]:
    """
    Create a folder for all the on-the-fly generated fixtures
    """
    tmp_fixture_dir = Path("tmp_fixtures/")
    tmp_fixture_dir.mkdir()
    yield tmp_fixture_dir
    shutil.rmtree(tmp_fixture_dir, ignore_errors=True)


@fixture()
def fixtures_dir() -> Path:
    return (Path(__file__).parent / "fixtures").absolute()


@fixture
def tmp_dir_with_content(tmp_dir: Path, fixtures_dir: Path) -> Iterator[Path]:
    tileset_folder = tmp_dir / "simple_xyz"
    convert(fixtures_dir / "simple.xyz", outfolder=tileset_folder, overwrite=True)
    yield tileset_folder


@fixture
def tileset_pnts_path_1(tmp_dir: Path, fixtures_dir: Path) -> Iterator[Path]:
    tileset_folder = tmp_dir / "1"
    convert(
        fixtures_dir / "with_srs_3857.las",
        crs_out=CRS.from_epsg(3950),
        outfolder=tileset_folder,
        extra_fields=["intensity", "raw_classification"],
    )
    yield tileset_folder / "tileset.json"


@fixture
def tileset_pnts_1(tileset_pnts_path_1: Path) -> TileSet:
    return TileSet.from_file(tileset_pnts_path_1)


@fixture
def tileset_pnts_path_2(tmp_dir: Path, fixtures_dir: Path) -> Iterator[Path]:
    tileset_folder = tmp_dir / "2"
    convert(
        fixtures_dir / "with_srs_3950.las",
        outfolder=tileset_folder,
        extra_fields=["intensity", "raw_classification"],
    )
    yield tileset_folder / "tileset.json"


@fixture
def tileset_pnts_2(tileset_pnts_path_2: Path) -> TileSet:
    tileset = TileSet.from_file(tileset_pnts_path_2)
    tileset.root_tile.get_or_fetch_content(tileset_pnts_path_2.parent)
    return tileset


@fixture
def tileset_ifc_path_1(tmp_dir: Path, fixtures_dir: Path) -> Iterator[Path]:
    tileset_folder = tmp_dir / "simple1_ifc"
    convert(fixtures_dir / "simple1.ifc", outfolder=tileset_folder, overwrite=True)
    yield tileset_folder / "tileset.json"


@fixture
def tileset_ifc_1(tileset_ifc_path_1: Path) -> TileSet:
    return TileSet.from_file(tileset_ifc_path_1)


@fixture
def tileset_ifc_path_2(tmp_dir: Path, fixtures_dir: Path) -> Iterator[Path]:
    tileset_folder = tmp_dir / "simple2_ifc"
    convert(fixtures_dir / "simple2.ifc", outfolder=tileset_folder, overwrite=True)
    yield tileset_folder / "tileset.json"


@fixture
def tileset_ifc_2(tileset_ifc_path_2: Path) -> TileSet:
    return TileSet.from_file(tileset_ifc_path_2)


@fixture
def ply_filepath(fixtures_dir: Path) -> Path:
    return fixtures_dir / "simple.ply"


@fixture
def buggy_ply_filepath(fixtures_dir: Path) -> Path:
    return fixtures_dir / "buggy.ply"


@fixture
def ply_with_extra_fields_filepath(fixtures_dir: Path) -> Path:
    return fixtures_dir / "simple_with_classification_and_intensity.ply"


@fixture
def ply_with_extra_fields_as_f4_filepath(fixtures_dir: Path) -> Path:
    return fixtures_dir / "simple_with_classification_and_intensity_f4.ply"


@fixture
def ply_with_intensity_filepath(fixtures_dir: Path) -> Path:
    return fixtures_dir / "simple_with_intensity.ply"


@fixture
def ply_with_classification_filepath(fixtures_dir: Path) -> Path:
    return fixtures_dir / "simple_with_classification.ply"


@fixture(scope="session")
def ply_big_with_one_additional_field_filepath(tmp_fixture_dir: Path) -> Iterator[Path]:
    f = tmp_fixture_dir / "big_with_one_additional_field.ply"
    # with a non-trivial amount of vertices
    vertex_array = []
    for i in range(100_001):
        vertex_array.append((i / 10_000, -i / 10_000, 0, i / 1000))
    vertex = np.array(
        vertex_array, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("field", "f4")]
    )
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(f)
    yield f
    # tmp_fixture_dir will get removed entirely at the end of the test session


@fixture
def northing_easting_ordering_2326_xyz(fixtures_dir: Path) -> Path:
    # "correct" ordering, the one mandated by the definition
    # NOTE: this is the victoria peak in Hong Kong
    return fixtures_dir / "northing_easting_ordering_2326.xyz"


@fixture
def easting_northing_ordering_2326_xyz(fixtures_dir: Path) -> Path:
    # traditional gis ordering: easting northing
    # NOTE: this is the victoria peak in Hong Kong
    return fixtures_dir / "easting_northing_ordering_2326.xyz"


@fixture
def ripple_filepath(fixtures_dir: Path) -> Path:
    return fixtures_dir / "ripple.las"


@fixture(params=["wrongname", "vertex"])
def buggy_ply_data(request) -> dict[str, Any]:  # type: ignore [no-untyped-def]
    """This ply data does not contain any 'vertex' element!"""
    types = [("x", np.float32, (5,)), ("y", np.float32, (5,)), ("z", np.float32, (5,))]
    data = [(np.random.sample(5), np.random.sample(5), np.random.sample(5))]
    if request.param == "wrongname":
        arr = np.array(data, dtype=np.dtype(types))
    else:
        arr = np.array([data[0][:2]], np.dtype(types[:2]))
    ply_item = plyfile.PlyElement.describe(data=arr, name=request.param)
    ply_data = plyfile.PlyData(elements=[ply_item])
    return {
        "data": ply_data,
        "msg": "vertex" if request.param == "wrongname" else "x, y, z",
    }


@fixture
def node() -> Node:
    bbox = np.array([[0, 0, 0], [2, 2, 2]])
    return Node(
        b"noeud",
        bbox,
        compute_spacing(bbox),
        True,
        [
            ExtraFieldsDescription(name="classification", dtype=np.dtype(np.uint8)),
            ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.uint8)),
        ],
    )


@fixture
def node_position_only() -> Node:
    bbox = np.array([[0, 0, 0], [2, 2, 2]])
    return Node(b"noeud", bbox, compute_spacing(bbox), False, [])


@fixture
def grid(node: Node) -> Grid:
    return Grid(node.spacing, node.include_rgb, node.extra_fields)


@fixture
def grid_position_only(node_position_only: Node) -> Grid:
    return Grid(
        node_position_only.spacing,
        node_position_only.include_rgb,
        node_position_only.extra_fields,
    )


@fixture
def tileset() -> TileSet:
    """
    Programmatically define a tileset sample encountered in the
    TileSet json header specification cf
    https://github.com/AnalyticalGraphicsInc/3d-tiles/tree/master/specification#tileset-json
    :return: a TileSet object.
    """
    tile_set = TileSet()
    bounding_volume = BoundingVolumeBox()
    bounding_volume.set_from_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    root_tile = Tile(geometric_error=3.14159, bounding_volume=bounding_volume)
    # Setting the mode to the default mode does not really change things.
    # The following line is thus just here ot test the "callability" of
    # set_refine_mode():
    root_tile.set_refine_mode("ADD")
    tile_set.root_tile = root_tile

    extension = MockExtension("Test")
    tile_set.extensions[extension.name] = extension
    tile_set.extensions_used.add(extension.name)

    return tile_set


@fixture
def tileset_on_disk_with_sub_tileset_path(
    tmp_dir: Path, fixtures_dir: Path
) -> Iterator[Path]:
    convert(fixtures_dir / "simple.xyz", outfolder=tmp_dir, overwrite=True)

    sub_tileset_path = tmp_dir / "tileset.json"
    main_tileset_path = tmp_dir / "upper_tileset.json"
    sub_tileset = TileSet.from_file(sub_tileset_path)

    tileset = TileSet()
    tileset.root_tile.content_uri = Path("tileset.json")
    tileset.root_tile.bounding_volume = copy.deepcopy(
        sub_tileset.root_tile.bounding_volume
    )
    tileset.root_tile.transform = copy.deepcopy(sub_tileset.root_tile.transform)
    tileset.write_as_json(main_tileset_path)

    yield main_tileset_path


@fixture
def batch_table_hierarchy_reference_file(fixtures_dir: Path) -> Path:
    return fixtures_dir / "batch_table_hierarchy_reference_sample.json"


@fixture
def batch_table_hierarchy_with_indexes() -> BatchTableHierarchy:
    """
    Programmatically define the reference sample encountered in the
    BTH specification cf
    https://github.com/AnalyticalGraphicsInc/3d-tiles/tree/master/extensions/3DTILES_batch_table_hierarchy#batch-table-json-schema-updates
    :return: the sample as BatchTableHierarchy object.
    """
    bth = BatchTableHierarchy()

    wall_class = bth.add_class("Wall", ["color"])
    wall_class.add_instance({"color": "white"}, [6])
    wall_class.add_instance({"color": "red"}, [6, 10, 11])
    wall_class.add_instance({"color": "yellow"}, [7, 11])
    wall_class.add_instance({"color": "gray"}, [7])
    wall_class.add_instance({"color": "brown"}, [8])
    wall_class.add_instance({"color": "black"}, [8])

    building_class = bth.add_class("Building", ["name", "address"])
    building_class.add_instance({"name": "unit29", "address": "100 Main St"}, [10])
    building_class.add_instance({"name": "unit20", "address": "102 Main St"}, [10])
    building_class.add_instance({"name": "unit93", "address": "104 Main St"}, [9])

    owner_class = bth.add_class("Owner", ["type", "id"])
    owner_class.add_instance({"type": "city", "id": 1120})
    owner_class.add_instance({"type": "resident", "id": 1250})
    owner_class.add_instance({"type": "commercial", "id": 6445})
    return bth


@fixture
def batch_table_hierarchy_with_instances() -> BatchTableHierarchy:
    bth = BatchTableHierarchy()

    wall_class = bth.add_class("Wall", ["color"])
    building_class = bth.add_class("Building", ["name", "address"])
    owner_class = bth.add_class("Owner", ["type", "id"])

    owner_city = owner_class.add_instance({"type": "city", "id": 1120})
    owner_resident = owner_class.add_instance({"type": "resident", "id": 1250})
    owner_commercial = owner_class.add_instance({"type": "commercial", "id": 6445})

    building_29 = building_class.add_instance(
        {"name": "unit29", "address": "100 Main St"}, [owner_resident]
    )
    building_20 = building_class.add_instance(
        {"name": "unit20", "address": "102 Main St"}, [owner_resident]
    )
    building_93 = building_class.add_instance(
        {"name": "unit93", "address": "104 Main St"}, [owner_city]
    )

    wall_class.add_instance({"color": "white"}, [building_29])
    wall_class.add_instance(
        {"color": "red"}, [building_29, owner_resident, owner_commercial]
    )
    wall_class.add_instance({"color": "yellow"}, [building_20, owner_commercial])
    wall_class.add_instance({"color": "gray"}, [building_20])
    wall_class.add_instance({"color": "brown"}, [building_93])
    wall_class.add_instance({"color": "black"}, [building_93])
    return bth


@fixture
def clockwise_star(fixtures_dir: Path) -> PolygonType:
    with open(fixtures_dir / "star_clockwise.geojson") as f:
        star_geo = json.load(f)
        coords: PolygonType = star_geo["features"][0]["geometry"]["coordinates"]
        # triangulate expects the coordinates to be numpy array
        polygon = coords[0]
        for i in range(len(polygon)):
            polygon[i] = np.array(polygon[i], dtype=np.float32)
        # triangulate implicitly use wkb format, which is not self-closing
        del polygon[-1]
        return coords


@fixture
def counterclockwise_star(fixtures_dir: Path) -> PolygonType:
    with open(fixtures_dir / "star_counterclockwise.geojson") as f:
        star_geo = json.load(f)
        coords: PolygonType = star_geo["features"][0]["geometry"]["coordinates"]
        # triangulate expects the coordinates to be numpy array
        polygon = coords[0]
        for i in range(len(polygon)):
            polygon[i] = np.array(polygon[i], dtype=np.float32)
        # triangulate implicitly use wkb format, which is not self-closing
        del polygon[-1]
        return coords


@fixture
def counterclockwise_zx_star(fixtures_dir: Path) -> PolygonType:
    with open(fixtures_dir / "star_zx_counter_clockwise.geojson") as f:
        star_geo = json.load(f)
        coords: PolygonType = star_geo["features"][0]["geometry"]["coordinates"]
        # triangulate expects the coordinates to be numpy array
        polygon = coords[0]
        for i in range(len(polygon)):
            polygon[i] = np.array(polygon[i], dtype=np.float32)
        # triangulate implicitly use wkb format, which is not self-closing
        del polygon[-1]
        return coords


@fixture
def big_poly(fixtures_dir: Path) -> PolygonType:
    with open(fixtures_dir / "big_polygon_counter_clockwise.geojson") as f:
        big_poly = json.load(f)
        coords: PolygonType = big_poly["features"][0]["geometry"]["coordinates"]
        # triangulate expects the coordinates to be numpy array
        polygon = coords[0]
        for i in range(len(polygon)):
            polygon[i] = np.array(polygon[i], dtype=np.float32)
        # triangulate implicitly use wkb format, which is not self-closing
        del polygon[-1]
        return coords


@fixture
def complex_polygon() -> PolygonType:
    # tricky polygon 1:
    # 0x---------x 4
    #   \        |
    #    \       |
    #   1 x      |
    #    /       |
    #   /        |
    # 2x---------x 3
    # the first few vertices seems to indicate an inverse winding order
    return [
        [
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0.5, 0.5, 0], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 0], dtype=np.float32),
            np.array([1, 1, 0], dtype=np.float32),
        ]
    ]


@fixture
def xyz() -> npt.NDArray[np.float32]:
    return np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0.5],
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0.75, 0.25, 0],
            [0.75, 0.25, 0.5],
            [0.25, 0.75, 1],
        ],
        dtype=np.float32,
    )


@fixture
def z_up_matrix() -> npt.NDArray[np.float32]:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
    )


@fixture
def dummy_matrix() -> list[float]:
    # fmt: off
    return [
        0, 0, 0,
        1, 0, 0,
        0, 2, 0,
        0, 0, 3.4,
    ]
    # fmt: on


@fixture
def bounding_volume_box_sample(dummy_matrix: list[np.float32]) -> BoundingVolumeBox:
    bounding_volume_box = BoundingVolumeBox()
    bounding_volume_box.set_from_list(dummy_matrix)
    return bounding_volume_box


@fixture
def complex_bounding_volume_box() -> BoundingVolumeBox:
    """
    A more complex bounding box with the center not in [0, 0, 0] and axes not aligned with world axis
    """
    # fmt: off
    return BoundingVolumeBox.from_list([
        1, 2, 3,
        1, 1, 0,
        2, -2, 0,
        0, 0, 5
    ])
    # fmt: on
