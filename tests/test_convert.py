import json
import multiprocessing
import os
import shutil
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from time import sleep
from unittest.mock import patch

import laspy
import numpy as np
import plyfile
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pyproj import CRS
from pytest import RaisesExc, mark, raises

from py3dtiles.convert import Converter, convert
from py3dtiles.exceptions import (
    SrsInMissingException,
    SrsInMixinException,
    TilerNotFoundException,
)
from py3dtiles.reader.ply_reader import create_plydata_with_renamed_property
from py3dtiles.tilers.base_tiler import Tiler
from py3dtiles.tilers.base_tiler.shared_metadata import SharedMetadata
from py3dtiles.tilers.base_tiler.tiler_worker import TilerWorker
from py3dtiles.tileset import TileSet, number_of_points_in_tileset
from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.content import Pnts
from py3dtiles.tileset.tile import Tile


def test_convert(tmp_dir: Path, fixtures_dir: Path) -> None:
    path = fixtures_dir / "ripple.las"
    convert(path, outfolder=tmp_dir)

    # basic asserts
    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [0.0, 0.0, 0.0, 5.0, 0, 0, 0, 5.0, 0, 0, 0, 0.8832]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box
    assert tileset["root"]["transform"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.2167949676513672,
        1.0,
    ]
    # the preview in on the root of outfolder
    assert tileset["root"]["content"]["uri"] == "preview.pnts"
    assert Path(tmp_dir, tileset["root"]["content"]["uri"]).exists()
    # but all the others are inside the "points" folder
    assert tileset["root"]["children"][0]["content"]["uri"] == "points/r.pnts"
    assert Path(tmp_dir, tileset["root"]["children"][0]["content"]["uri"]).exists()
    children = tileset["root"]["children"][0]["children"]
    for child in children:
        pnts_path = child["content"]["uri"]
        assert pnts_path.startswith("points/")
        assert pnts_path.endswith(".pnts")
        assert Path(tmp_dir, pnts_path).exists()

    assert Path(tmp_dir, "points", "r.pnts").exists()
    assert Path(tmp_dir, "points", "r0.pnts").exists()

    with laspy.open(path) as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_ifc(tmp_dir: Path, tileset_ifc_1: TileSet) -> None:
    # basic asserts
    assert tileset_ifc_1.root_uri is not None
    tileset_path = tileset_ifc_1.root_uri / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 3.5434]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box
    assert_array_equal(
        tileset["root"]["transform"],
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            6.0,
            5.0,
            2.543375790119171,
            1.0,
        ],
    )

    # the preview in on the root of outfolder
    # at the moment, there is no preview tiles for ifc tilesets
    assert "content" not in tileset["root"]

    # the root tile is the IfcProject, no content as well at the moment
    assert "content" not in tileset["root"]["children"][0]

    # assert the other tiles
    children = tileset["root"]["children"][0]["children"]
    for child in children:
        content_path = child["content"]["uri"]
        assert content_path.startswith("ifc/")
        assert content_path.endswith(".b3dm")
        assert Path(tileset_ifc_1.root_uri, content_path).exists()


def test_convert_with_prune(tmp_dir: Path, fixtures_dir: Path) -> None:
    # This file has 1 point at (-2, -2, -2) and 20001 at (1, 1, 1)
    # like this, it triggers the prune mechanism
    laz_path = fixtures_dir / "stacked_points.las"

    convert(
        laz_path,
        outfolder=tmp_dir,
        jobs=1,
        rgb=False,  # search bound cases by disabling rgb export
    )

    # basic asserts
    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    assert tileset["root"]["transform"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -0.5,
        -0.5,
        -0.5,
        1.0,
    ]
    expecting_box = [0, 0, 0, 1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5]
    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == expecting_box

    assert Path(tmp_dir, "points", "r.pnts").exists()
    assert Path(tmp_dir, "points", "r0.pnts").exists()

    with laspy.open(laz_path) as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_without_srs(tmp_dir: Path, fixtures_dir: Path) -> None:
    with raises(SrsInMissingException):
        convert(
            fixtures_dir / "without_srs.las",
            outfolder=tmp_dir,
            crs_out=CRS.from_epsg(4978),
            jobs=1,
        )
    assert not tmp_dir.exists()

    convert(
        fixtures_dir / "without_srs.las",
        outfolder=tmp_dir,
        crs_in=CRS.from_epsg(3949),
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )

    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    assert_array_almost_equal(
        tileset["root"]["transform"],
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            4203111.466352918,
            171019.2566206994,
            4778277.619872708,
            1.0,
        ],
    )

    box = [round(value, 4) for value in tileset["root"]["boundingVolume"]["box"]]
    assert box == [0.0, 0.0, 0.0, 0.5057, 0.0, 0.0, 0.0, 0.9801, 0.0, 0.0, 0.0, 0.5432]

    assert Path(tmp_dir, "points", "r.pnts").exists()

    with laspy.open(fixtures_dir / "without_srs.las") as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)

    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((187, 187, 187), dtype=np.uint8))


def test_convert_las_color_scale(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(
        fixtures_dir / "without_srs.las",
        outfolder=tmp_dir,
        jobs=1,
    )

    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((187, 187, 187), dtype=np.uint8))
    convert(
        fixtures_dir / "without_srs.las",
        overwrite=True,
        color_scale=1.1,
        outfolder=tmp_dir,
        jobs=1,
    )
    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # it should clamp to 255
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((206, 206, 206), dtype=np.uint8))

    convert(
        fixtures_dir / "without_srs.las",
        overwrite=True,
        color_scale=1.5,
        outfolder=tmp_dir,
        jobs=1,
    )
    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # it should clamp to 255
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((255, 255, 255), dtype=np.uint8))


def test_convert_with_srs(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(
        fixtures_dir / "with_srs_3857.las",
        outfolder=tmp_dir,
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )

    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    assert_array_almost_equal(
        tileset["root"]["transform"],
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -435960.9784218713,
            -5438487.100036068,
            3292675.726167237,
            1.0,
        ],
    )
    assert_array_almost_equal(
        tileset["root"]["children"][0]["transform"],
        [
            99.67987521469695,
            -7.9951492419151196,
            0.008110606960175354,
            0.0,
            4.123612711579332,
            51.49818036951855,
            85.6208691665381,
            0.0,
            -6.849693087091023,
            -85.34644109292454,
            51.66301092062502,
            0.0,
            -535.4490030509769,
            -211.56234651152045,
            -452.35058756079525,
            1.0,
        ],
    )
    expecting_box = [
        5.1686,
        5.1834,
        0.1646,
        5.1684,
        0.0,
        0.0,
        0.0,
        5.1834,
        0.0,
        0.0,
        0.0,
        0.1952,
    ]
    box = [
        round(value, 4)
        for value in tileset["root"]["children"][0]["boundingVolume"]["box"]
    ]
    assert box == expecting_box

    assert Path(tmp_dir, "points", "r.pnts").exists()

    with laspy.open(fixtures_dir / "with_srs_3857.las") as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_simple_xyz(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(
        fixtures_dir / "simple.xyz",
        outfolder=tmp_dir,
        crs_in=CRS.from_epsg(3857),
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    xyz_point_count = 0
    with open(fixtures_dir / "simple.xyz") as f:
        while line := f.readline():
            xyz_point_count += 1 if line != "" else 0

    tileset_path = tmp_dir / "tileset.json"
    assert xyz_point_count == number_of_points_in_tileset(tileset_path)

    with tileset_path.open() as f:
        tileset = json.load(f)
    expecting_box = [0.3916, 0.3253, -0.0001, 0.39, 0, 0, 0, 0.3099, 0, 0, 0, 0.0001]
    box = [
        round(value, 4)
        for value in tileset["root"]["children"][0]["boundingVolume"]["box"]
    ]
    assert box == expecting_box


def test_convert_xyz_rgb_i_c(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(
        fixtures_dir / "simple_with_irgb_and_classification.csv",
        outfolder=tmp_dir,
        jobs=1,
        extra_fields=["intensity", "classification"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    xyz_point_count = -1  # compensate for header line
    with open(fixtures_dir / "simple_with_irgb_and_classification.csv") as f:
        while line := f.readline():
            xyz_point_count += 1 if line != "" else 0

    tileset_path = tmp_dir / "tileset.json"
    assert xyz_point_count == number_of_points_in_tileset(tileset_path)

    tileset = TileSet.from_file(tileset_path)
    tile_content = Pnts.from_file(tmp_dir / "points" / "r.pnts")
    ft_body = tile_content.body.feature_table.body
    # assert position
    local_coords = tileset.root_tile.children[0].transform_coords(
        ft_body.position.reshape(-1, 3)
    )
    world_coords = tileset.root_tile.transform_coords(local_coords)

    assert_array_equal(
        world_coords,
        np.array(
            [
                [281345.9, 369123.25, 0.0],
                [281345.12, 369123.47, 0.0],
                [281345.56, 369123.88, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert ft_body.color is not None
    assert_array_equal(ft_body.color, [0, 0, 200, 10, 0, 0, 0, 10, 0])
    # batch table
    bt = tile_content.body.batch_table
    assert_array_equal(bt.get_binary_property("intensity"), [3, 1, 2])
    assert_array_equal(bt.get_binary_property("classification"), [22, 21, 22])


def test_convert_xyz_rgb_i_c_with_srs(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(
        fixtures_dir / "simple_with_irgb_and_classification.csv",
        outfolder=tmp_dir,
        crs_in=CRS.from_epsg(28992),
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    xyz_point_count = -1  # compensate for header line
    with open(fixtures_dir / "simple_with_irgb_and_classification.csv") as f:
        while line := f.readline():
            xyz_point_count += 1 if line != "" else 0

    tileset_path = tmp_dir / "tileset.json"
    assert xyz_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_xyz_with_rgb(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(fixtures_dir / "simple_with_rgb.xyz", outfolder=tmp_dir)

    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((10, 0, 0), dtype=np.uint8))

    tile2 = Pnts.from_file(tmp_dir / "points" / "r4.pnts")
    assert tile2.body.feature_table.nb_points() == 1
    pt2_color = tile2.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt2_color is None:
        raise RuntimeError("pt2_color shouldn't be None.")
    assert_array_equal(pt2_color, np.array((0, 0, 200), dtype=np.uint8))

    tile3 = Pnts.from_file(tmp_dir / "points" / "r6.pnts")
    assert tile3.body.feature_table.nb_points() == 1
    pt3_color = tile3.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt3_color is None:
        raise RuntimeError("pt3_color shouldn't be None.")
    assert_array_equal(pt3_color, np.array((0, 10, 0), dtype=np.uint8))


def test_convert_xyz_with_rgb_color_scale(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(fixtures_dir / "simple_with_rgb.xyz", outfolder=tmp_dir, color_scale=1.5)

    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((15, 0, 0), dtype=np.uint8))

    tile2 = Pnts.from_file(tmp_dir / "points" / "r4.pnts")
    assert tile2.body.feature_table.nb_points() == 1
    pt2_color = tile2.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt2_color is None:
        raise RuntimeError("pt2_color shouldn't be None.")
    assert_array_equal(pt2_color, np.array((0, 0, 255), dtype=np.uint8))

    tile3 = Pnts.from_file(tmp_dir / "points" / "r6.pnts")
    assert tile3.body.feature_table.nb_points() == 1
    pt3_color = tile3.body.feature_table.get_feature_color_at(0)
    # Note the first point is taken as offset base
    if pt3_color is None:
        raise RuntimeError("pt3_color shouldn't be None.")
    assert_array_equal(pt3_color, np.array((0, 15, 0), dtype=np.uint8))


def test_convert_ply(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(fixtures_dir / "simple.ply", outfolder=tmp_dir, jobs=1)
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    expected_point_count = 22300
    tileset_path = tmp_dir / "tileset.json"
    assert expected_point_count == number_of_points_in_tileset(tileset_path)

    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [
        4.5437,
        5.5984,
        1.1842,
        4.5437,
        0.0,
        0.0,
        0.0,
        5.5984,
        0.0,
        0.0,
        0.0,
        1.1842,
    ]
    box = [
        round(value, 4)
        for value in tileset["root"]["children"][0]["boundingVolume"]["box"]
    ]
    assert box == expecting_box

    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 5293
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 0), dtype=np.uint8))


def test_convert_ply_with_color(tmp_dir: Path, fixtures_dir: Path) -> None:
    # 8 bits color
    convert(fixtures_dir / "simple_with_8_bits_colors.ply", outfolder=tmp_dir, jobs=1)
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    expected_point_count = 4
    tileset_path = tmp_dir / "tileset.json"
    assert expected_point_count == number_of_points_in_tileset(tileset_path)

    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 128, 0), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r3.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((10, 0, 0), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r5.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 20), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r6.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((40, 40, 40), dtype=np.uint8))

    # 16 bits colors
    # every value should be divided by 256
    convert(
        fixtures_dir / "simple_with_16_bits_colors.ply",
        outfolder=tmp_dir,
        jobs=1,
        overwrite=True,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    expected_point_count = 4
    tileset_path = tmp_dir / "tileset.json"
    assert expected_point_count == number_of_points_in_tileset(tileset_path)

    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 0), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r3.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((1, 0, 0), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r5.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 4), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r6.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((255, 255, 255), dtype=np.uint8))


def test_convert_ply_with_color_scale(tmp_dir: Path, fixtures_dir: Path) -> None:
    # 8 bits color
    convert(
        fixtures_dir / "simple_with_8_bits_colors.ply",
        outfolder=tmp_dir,
        jobs=1,
        color_scale=3,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    tile1 = Pnts.from_file(tmp_dir / "points" / "r0.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 255, 0), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r3.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((30, 0, 0), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r5.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((0, 0, 60), dtype=np.uint8))

    tile1 = Pnts.from_file(tmp_dir / "points" / "r6.pnts")
    assert tile1.body.feature_table.nb_points() == 1
    pt1_color = tile1.body.feature_table.get_feature_color_at(0)
    if pt1_color is None:
        raise RuntimeError("pt1_color shouldn't be None.")
    assert_array_equal(pt1_color, np.array((120, 120, 120), dtype=np.uint8))


def test_convert_ply_with_wrong_classification(
    tmp_dir: Path, fixtures_dir: Path
) -> None:
    # Buggy feature name, but conversion should still pass
    convert(
        fixtures_dir / "simple.ply",
        outfolder=tmp_dir,
        jobs=1,
        extra_fields=["classification"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    for py3dt_file in tmp_dir.iterdir():
        if py3dt_file.suffix != ".pnts":
            continue
        tile_content = Pnts.from_file(py3dt_file)
        # no other file, so no classification
        assert "classification" not in tile_content.body.batch_table.header.data

    convert(
        [fixtures_dir / "simple.ply", fixtures_dir / "simple_with_classification.ply"],
        outfolder=tmp_dir,
        jobs=1,
        overwrite=True,
        extra_fields=["classification"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()
    classification: list[int] = []
    for py3dt_file in (tmp_dir / "points").iterdir():
        if py3dt_file.suffix != ".pnts":
            continue
        tile_content = Pnts.from_file(py3dt_file)
        classification.extend(
            tile_content.body.batch_table.get_binary_property("classification")
        )
    # the file simple.ply has no classification (will contribute the 0)
    # the file simple_with_classification.ply have 1s and 2s as classifications
    assert np.array_equal(
        np.unique(classification), np.array([0, 1, 2], dtype=np.uint8)
    )


def test_convert_ply_with_good_classification(
    tmp_dir: Path, fixtures_dir: Path
) -> None:
    EXPECTED_LABELS = np.array([-1, 0, 1, 2], dtype=np.int32)
    # Change the classification property name in the tested .ply file
    ply_data = plyfile.PlyData.read(fixtures_dir / "simple.ply")
    ply_data = create_plydata_with_renamed_property(ply_data, "label", "classification")
    modified_ply_filename = fixtures_dir / "modified.ply"
    ply_data.write(modified_ply_filename)
    # Valid feature name, classification is preserved.
    convert(
        modified_ply_filename,
        outfolder=tmp_dir,
        jobs=1,
        extra_fields=["classification"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    tileset_labels = np.array((), dtype=np.uint8)
    for py3dt_file in (tmp_dir / "points").iterdir():
        if py3dt_file.suffix != ".pnts":
            continue
        tile_content = Pnts.from_file(py3dt_file)
        assert "classification" in tile_content.body.batch_table.header.data
        # classification is OK for each pnts
        pnts_labels = tile_content.body.batch_table.get_binary_property(
            "classification"
        )
        tileset_labels = np.unique(np.append(tileset_labels, pnts_labels))
    # Every label is encountered in the global tileset
    assert np.array_equal(tileset_labels, EXPECTED_LABELS)
    # Clean the test directory
    modified_ply_filename.unlink()


def test_convert_ply_with_intensity(tmp_dir: Path, fixtures_dir: Path) -> None:
    # Valid feature name, intensity is preserved.
    convert(
        fixtures_dir / "simple_with_intensity.ply",
        outfolder=tmp_dir,
        jobs=1,
        extra_fields=["intensity"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    tile_content = Pnts.from_file(tmp_dir / "points" / "r.pnts")
    assert_array_equal(
        tile_content.body.feature_table.body.position,
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    )
    assert "intensity" in tile_content.body.batch_table.header.data
    assert_array_equal(
        [80, 129, 15, 90],
        tile_content.body.batch_table.get_binary_property("intensity"),
    )


def test_convert_ply_with_classification_and_intensity(
    tmp_dir: Path, ply_with_extra_fields_filepath: Path
) -> None:
    # Valid feature name, intensity is preserved.
    convert(
        ply_with_extra_fields_filepath,
        outfolder=tmp_dir,
        jobs=1,
        extra_fields=["intensity", "classification"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    tile_content = Pnts.from_file(tmp_dir / "points" / "r.pnts")
    assert_array_equal(
        tile_content.body.feature_table.body.position,
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    )
    assert "classification" in tile_content.body.batch_table.header.data
    assert_array_equal(
        [1, 2, 2, 1],
        tile_content.body.batch_table.get_binary_property("classification"),
    )
    assert "intensity" in tile_content.body.batch_table.header.data
    assert_array_equal(
        [-80, 129129.19, 15.3, 90.2],
        tile_content.body.batch_table.get_binary_property("intensity"),
    )


def test_convert_ply_with_classification_and_intensity_f4(
    tmp_dir: Path, ply_with_extra_fields_as_f4_filepath: Path
) -> None:
    # we don't support arbitrary precision in intensity yet, but the conversion should succeed with warnings
    convert(
        ply_with_extra_fields_as_f4_filepath,
        outfolder=tmp_dir,
        jobs=1,
        extra_fields=["intensity", "classification"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    tile_content = Pnts.from_file(tmp_dir / "points" / "r.pnts")
    assert_array_equal(
        tile_content.body.feature_table.body.position,
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    )
    assert "classification" in tile_content.body.batch_table.header.data
    assert_array_equal(
        [1, 2, 2, 1],
        tile_content.body.batch_table.get_binary_property("classification"),
    )
    assert "intensity" in tile_content.body.batch_table.header.data
    assert_array_almost_equal(
        tile_content.body.batch_table.get_binary_property("intensity"),
        np.array([80.1, -129.0, -15.0, 90.4], dtype=np.float32),
    )

    # without intensity
    convert(
        ply_with_extra_fields_as_f4_filepath,
        outfolder=tmp_dir,
        jobs=1,
        overwrite=True,
        extra_fields=["classification"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    tile_content = Pnts.from_file(tmp_dir / "points" / "r.pnts")
    assert_array_equal(
        tile_content.body.feature_table.body.position,
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    )
    assert "classification" in tile_content.body.batch_table.header.data
    assert_array_equal(
        [1, 2, 2, 1],
        tile_content.body.batch_table.get_binary_property("classification"),
    )
    assert "intensity" not in tile_content.body.batch_table.header.data


def test_convert_ply_big(
    tmp_dir: Path, ply_big_with_one_additional_field_filepath: Path
) -> None:
    # without anything
    convert(
        ply_big_with_one_additional_field_filepath,
        outfolder=tmp_dir,
        jobs=1,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()
    tileset = TileSet.from_file(tmp_dir / "tileset.json")
    tile_content = tileset.root_tile.get_or_fetch_content(tileset.root_uri)
    assert isinstance(tile_content, Pnts)
    # assert field is not present
    with raises(ValueError):
        tile_content.get_batch_table_binary_property("field")

    # with one additional field
    convert(
        ply_big_with_one_additional_field_filepath,
        outfolder=tmp_dir,
        jobs=1,
        overwrite=True,
        extra_fields=["field"],
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()
    tileset = TileSet.from_file(tmp_dir / "tileset.json")
    tile_content = tileset.root_tile.get_or_fetch_content(tileset.root_uri)
    assert isinstance(tile_content, Pnts)
    # get world coords
    points = tile_content.get_points(tileset.root_tile.transform)

    # we constructed the ply so that there's a x10 factor between x and field
    fields = tile_content.get_batch_table_binary_property("field")
    assert fields is not None

    # fields = tile_content.body.batch_table.header.data['field']
    assert_array_almost_equal(points.positions[:, 0], fields / 10)


def test_convert_mix_las_xyz(tmp_dir: Path, fixtures_dir: Path) -> None:
    convert(
        [fixtures_dir / "simple.xyz", fixtures_dir / "with_srs_3857.las"],
        outfolder=tmp_dir,
        crs_out=CRS.from_epsg(4978),
        jobs=1,
    )
    assert Path(tmp_dir, "tileset.json").exists()
    assert Path(tmp_dir, "points", "r.pnts").exists()

    xyz_point_count = 0
    with open(fixtures_dir / "simple.xyz") as f:
        while line := f.readline():
            xyz_point_count += 1 if line != "" else 0

    with laspy.open(fixtures_dir / "with_srs_3857.las") as f:
        las_point_count = f.header.point_count

    tileset_path = tmp_dir / "tileset.json"
    assert xyz_point_count + las_point_count == number_of_points_in_tileset(
        tileset_path
    )

    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [
        3415.9824,
        5513.0671,
        -20917.9692,
        44316.7617,
        0.0,
        0.0,
        0.0,
        14858.1809,
        0.0,
        0.0,
        0.0,
        1585.8657,
    ]
    box = [
        round(value, 4)
        for value in tileset["root"]["children"][0]["boundingVolume"]["box"]
    ]
    assert box == expecting_box


def test_convert_mix_input_crs(tmp_dir: Path, fixtures_dir: Path) -> None:
    with raises(SrsInMixinException):
        convert(
            [
                fixtures_dir / "with_srs_3950.las",
                fixtures_dir / "with_srs_3857.las",
            ],
            outfolder=tmp_dir,
            crs_out=CRS.from_epsg(4978),
            jobs=1,
        )
    assert not tmp_dir.exists()

    with raises(SrsInMixinException):
        convert(
            [
                fixtures_dir / "with_srs_3950.las",
                fixtures_dir / "with_srs_3857.las",
            ],
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3432),
            crs_out=CRS.from_epsg(4978),
            jobs=1,
        )
    assert not tmp_dir.exists()

    convert(
        [fixtures_dir / "with_srs_3950.las", fixtures_dir / "with_srs_3857.las"],
        outfolder=tmp_dir,
        crs_in=CRS.from_epsg(3432),
        crs_out=CRS.from_epsg(4978),
        force_crs_in=True,
        jobs=1,
    )
    assert tmp_dir.exists()


@mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="'patch' function works only with the multiprocessing 'fork' method (not available on windows).",
)
def test_convert_xyz_exception_in_run(tmp_dir: Path, fixtures_dir: Path) -> None:
    with (
        patch("py3dtiles.reader.xyz_reader.run") as mock_run,
        raises(
            Exception,
            match="An exception occurred in a worker: builtins.Exception: Exception in run",
        ),
    ):
        # NOTE: this is intentionnally different from below, we are testing 2 different things
        # Here, we test that a very early fail wont block the run, and that it will terminate correctly
        mock_run.side_effect = Exception("Exception in run")
        convert(
            fixtures_dir / "simple.xyz",
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3857),
            crs_out=CRS.from_epsg(4978),
        )


@mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="'patch' function works only with the multiprocessing 'fork' method (not available on windows).",
)
def test_convert_las_exception_in_run(tmp_dir: Path, fixtures_dir: Path) -> None:
    with (
        patch("py3dtiles.reader.las_reader.run") as mock_run,
        raises(
            Exception,
            match="An exception occurred in a worker: builtins.Exception: Exception in run",
        ),
    ):

        def side_effect(*args):  # type: ignore
            sleep(1)
            raise Exception("Exception in run")

        mock_run.side_effect = side_effect
        convert(
            fixtures_dir / "with_srs_3857.las",
            outfolder=tmp_dir,
            crs_in=CRS.from_epsg(3857),
            crs_out=CRS.from_epsg(4978),
        )


def test_convert_export_folder_already_exists(
    tmp_dir: Path, fixtures_dir: Path
) -> None:
    assert not (tmp_dir / "tileset.json").exists()
    assert len(os.listdir(tmp_dir)) == 0

    # folder is empty, ok
    convert(
        fixtures_dir / "simple.xyz",
        outfolder=tmp_dir,
        jobs=1,
    )

    shutil.rmtree(tmp_dir)
    # folder will be created
    convert(
        fixtures_dir / "simple.xyz",
        outfolder=tmp_dir,
        jobs=1,
    )
    assert tmp_dir.exists()
    assert (tmp_dir / "tileset.json").exists()

    # now, subsequent conversion will fail
    with raises(
        FileExistsError, match=f"Folder '{tmp_dir}' already exists and is not empty."
    ):
        convert(
            fixtures_dir / "simple.xyz",
            outfolder=tmp_dir,
            jobs=1,
        )

    # but overwriting works
    convert(
        fixtures_dir / "simple.xyz",
        outfolder=tmp_dir,
        overwrite=True,
        jobs=1,
    )

    assert (tmp_dir / "tileset.json").exists()

    # finally, one file in folder is enough to bail
    shutil.rmtree(tmp_dir)
    tmp_dir.touch()
    with raises(
        FileExistsError,
        match=f"'{tmp_dir}' already exists and is not a directory. Not deleting it.",
    ):
        convert(
            fixtures_dir / "simple.xyz",
            outfolder=tmp_dir,
            jobs=1,
        )
    tmp_dir.unlink()


def test_convert_many_point_same_location(tmp_dir: Path) -> None:
    # This is how the file has been generated.
    xyz_path = tmp_dir / "pc_with_many_points_at_same_location.xyz"
    xyz_data = np.concatenate(
        (np.random.random((10000, 3)), np.repeat([[0, 0, 0]], repeats=20000, axis=0))
    )
    with xyz_path.open("w") as f:
        np.savetxt(f, xyz_data, delimiter=" ", fmt="%.10f")

    convert(xyz_path, outfolder=tmp_dir / "tiles")

    tileset_path = tmp_dir / "tiles" / "tileset.json"
    assert number_of_points_in_tileset(tileset_path) == 30000


@mark.parametrize(
    "rgb_bool,classif_bool",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_convert_rgb_classif(
    rgb_bool: bool, classif_bool: bool, tmp_dir: Path, fixtures_dir: Path
) -> None:
    expected_raise: nullcontext[None] | RaisesExc[ValueError]
    if not classif_bool:
        expected_raise = raises(
            ValueError, match="The property classification is not found"
        )
    else:
        # Ideally one does not raise if classification is required but pytest won't implement such
        # a feature (see https://github.com/pytest-dev/pytest/issues/1830). This solution is
        # suggested by a comment in this thread (see
        # https://github.com/pytest-dev/pytest/issues/1830#issuecomment-425653756).
        expected_raise = nullcontext()

    input_filepath = fixtures_dir / "simple_with_classification.ply"
    extra_fields = ["classification"] if classif_bool else []
    convert(input_filepath, rgb=rgb_bool, extra_fields=extra_fields, outfolder=tmp_dir)

    assert Path(tmp_dir, "points", "r.pnts").exists()

    ply_data = plyfile.PlyData.read(input_filepath)
    ply_point_count = ply_data.elements[0].count

    assert ply_point_count == number_of_points_in_tileset(tmp_dir / "tileset.json")

    tileset = TileSet.from_file(tmp_dir / "tileset.json")
    for tile_content in tileset.get_all_tile_contents():
        if isinstance(tile_content, TileSet):
            continue

        assert rgb_bool ^ (tile_content.body.feature_table.body.color is None)
        with expected_raise:
            bt_prop = tile_content.body.batch_table.get_binary_property(
                "classification"
            )
            assert len(bt_prop) > 0


def test_convert_without_threadpool(tmp_dir: Path, ripple_filepath: Path) -> None:
    convert(ripple_filepath, outfolder=tmp_dir, use_process_pool=False)

    # basic asserts
    tileset_path = tmp_dir / "tileset.json"
    with tileset_path.open() as f:
        tileset = json.load(f)

    expecting_box = [5.0, 5.0, 0.8832, 5.0, 0, 0, 0, 5.0, 0, 0, 0, 0.8832]
    box = [
        round(value, 4)
        for value in tileset["root"]["children"][0]["boundingVolume"]["box"]
    ]
    assert box == expecting_box

    assert Path(tmp_dir, "points", "r0.pnts").exists()

    with laspy.open(ripple_filepath) as f:
        las_point_count = f.header.point_count

    assert las_point_count == number_of_points_in_tileset(tileset_path)


def test_convert_crs_definition_ordering(
    tmp_dir: Path, northing_easting_ordering_2326_xyz: Path
) -> None:
    convert(
        northing_easting_ordering_2326_xyz,
        outfolder=tmp_dir,
        crs_in=CRS(2326),
        crs_out=CRS(3857),
    )
    tileset_path = tmp_dir / "tileset.json"
    assert number_of_points_in_tileset(tileset_path) == 2
    tileset = TileSet.from_file(tileset_path)
    # we assert on the offset of transform, it's the easiest really
    assert_array_almost_equal(
        tileset.root_tile.transform[:-1, 3], [12706529.0, 2544677.0, 536.0], decimal=0
    )


def test_convert_crs_traditional_ordering(
    tmp_dir: Path, easting_northing_ordering_2326_xyz: Path
) -> None:
    convert(
        easting_northing_ordering_2326_xyz,
        outfolder=tmp_dir,
        rgb=False,
        crs_in=CRS(2326),
        crs_out=CRS(3857),
        pyproj_always_xy=True,
    )
    tileset_path = tmp_dir / "tileset.json"
    assert number_of_points_in_tileset(tileset_path) == 2
    tileset = TileSet.from_file(tileset_path)
    # we assert on the offset of transform, it's the easiest really
    np.set_printoptions(suppress=True)
    assert_array_almost_equal(
        tileset.root_tile.transform[:-1, 3], [12706529.0, 2544677.0, 536.0], decimal=0
    )


class Metadata(SharedMetadata):
    pass


class Worker(TilerWorker[Metadata]):
    def execute(
        self, command: bytes, content: list[bytes]
    ) -> Iterator[Sequence[bytes]]:
        yield [b"WORK"]


class Tiler1(Tiler[Metadata, Worker]):
    name = "tiler1"

    def initialize(
        self, files: list[Path], working_dir: Path, out_folder: Path
    ) -> None:
        self.files = files
        self.get_tasks_called = False

    def supports(self, file: Path) -> bool:
        return file.suffix == ".1"

    def get_tasks(self) -> Iterator[tuple[bytes, list[bytes]]]:
        if not self.get_tasks_called:
            yield (b"command", [b"args"])
            self.get_tasks_called = True

    def process_message(self, message_type: bytes, message: list[bytes]) -> None:
        print(message_type, message)

    def get_worker(self) -> Worker:
        return Worker(Metadata())

    def get_root_tile(self, use_process_pool: bool = True) -> Tile:
        root_tile = Tile()
        root_tile.bounding_volume = BoundingVolumeBox.from_points(
            [np.array([0, 0, 0]), np.array([1, 1, 1])]
        )
        return root_tile


class Tiler2(Tiler[Metadata, Worker]):
    name = "tiler2"

    def initialize(
        self, files: list[Path], working_dir: Path, out_folder: Path
    ) -> None:
        self.files = files
        self.get_tasks_called = False

    def supports(self, file: Path) -> bool:
        return file.suffix == ".2"

    def get_tasks(self) -> Iterator[tuple[bytes, list[bytes]]]:
        if not self.get_tasks_called:
            yield (b"command", [b"args"])
            self.get_tasks_called = True

    def process_message(self, message_type: bytes, message: list[bytes]) -> None:
        print(message_type, message)

    def get_worker(self) -> Worker:
        return Worker(Metadata())

    def get_root_tile(self, use_process_pool: bool = True) -> Tile:
        root_tile = Tile()
        root_tile.bounding_volume = BoundingVolumeBox.from_points(
            [np.array([0, 0, 0]), np.array([2, 2, 2])]
        )
        return root_tile


def test_assign_file_to_tilers() -> None:
    converter = Converter([Tiler1(), Tiler2()])
    assert converter._assign_file_to_tilers(
        [Path("file.1"), Path("file.2"), Path("file2.2")]
    ) == {"tiler1": [Path("file.1")], "tiler2": [Path("file.2"), Path("file2.2")]}

    with raises(TilerNotFoundException):
        converter._assign_file_to_tilers(
            [Path("file.1"), Path("file.2"), Path("file2.2"), Path("file.3")]
        )


def test_convert_custom_tilers(tmp_dir: Path) -> None:
    tiler1 = Tiler1()
    converter = Converter([tiler1])

    converter.convert([Path("files.1")], tmp_dir / "custom_tilers")
    with raises(TilerNotFoundException):
        converter.convert([Path("files.2")], tmp_dir / "custom_tilers", overwrite=True)

    tiler2 = Tiler2()
    converter = Converter([tiler1, tiler2])
    converter.convert(
        [Path("files.2"), Path("files.1")], tmp_dir / "custom_tilers", overwrite=True
    )

    assert tiler1.files == [Path("files.1")]
    assert tiler2.files == [Path("files.2")]
