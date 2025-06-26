from pathlib import Path

from numpy.testing import assert_array_almost_equal, assert_array_equal

from py3dtiles.merger import merge, merge_from_files
from py3dtiles.tileset import BoundingVolumeBox, TileSet
from py3dtiles.tileset.content import Pnts, read_binary_tile_content


def test_merge_with_memory_tilesets(
    tmp_dir: Path, tileset_ifc_1: TileSet, tileset_ifc_2: TileSet
) -> None:
    merged_tileset = merge([tileset_ifc_1, tileset_ifc_2])

    assert len(merged_tileset.root_tile.get_all_children()) == 2

    # The main tilesets should link the tilesets
    assert merged_tileset.root_tile.children[0].tile_content is tileset_ifc_1
    assert merged_tileset.root_tile.children[1].tile_content is tileset_ifc_2

    # check if there is no URI
    for tile_child in merged_tileset.root_tile.get_all_children():
        assert tile_child.content_uri is None

    assert merged_tileset.geometric_error == max(
        tileset_ifc_1.root_tile.geometric_error, tileset_ifc_2.root_tile.geometric_error
    )
    assert merged_tileset.root_tile.geometric_error == max(
        tileset_ifc_1.root_tile.geometric_error, tileset_ifc_2.root_tile.geometric_error
    )

    # check if the bounding boxes are consistent
    for tile_child in merged_tileset.root_tile.get_all_children():
        sub_tileset = tile_child.get_or_fetch_content(tmp_dir)
        assert isinstance(sub_tileset, TileSet)

        sub_root_bounding_box = sub_tileset.root_tile.bounding_volume
        assert isinstance(sub_root_bounding_box, BoundingVolumeBox)
        sub_root_bounding_box.transform(sub_tileset.root_tile.transform)

        child_bounding_box = tile_child.bounding_volume
        assert isinstance(child_bounding_box, BoundingVolumeBox)
        child_bounding_box.transform(
            merged_tileset.root_tile.transform @ tile_child.transform
        )

        assert_array_almost_equal(sub_root_bounding_box._box, child_bounding_box._box)  # type: ignore [arg-type]

    # no preview for memory tilesets that are not pnts yet
    assert merged_tileset.root_tile.content_uri is None
    assert merged_tileset.root_tile.tile_content is None


def test_merge_with_pnts_content_with_memory_tilesets(
    tmp_dir: Path, tileset_pnts_1: TileSet, tileset_pnts_2: TileSet
) -> None:
    tileset_pnts_1.root_tile.get_or_fetch_content(tileset_pnts_1.root_uri)
    tileset_pnts_2.root_tile.get_or_fetch_content(tileset_pnts_1.root_uri)
    merged_tileset = merge([tileset_pnts_1, tileset_pnts_2])

    assert len(merged_tileset.root_tile.get_all_children()) == 2

    # The main tilesets should link the tilesets
    assert merged_tileset.root_tile.children[0].tile_content is tileset_pnts_1
    assert merged_tileset.root_tile.children[1].tile_content is tileset_pnts_2

    # check if there is no URI
    for tile_child in merged_tileset.root_tile.get_all_children():
        assert tile_child.content_uri is None

    assert merged_tileset.geometric_error == 1720.2520618394917
    assert merged_tileset.root_tile.geometric_error == 1720.2520618394917

    # check if the bounding boxes are consistent
    for tile_child in merged_tileset.root_tile.get_all_children():
        sub_tileset = tile_child.get_or_fetch_content(tmp_dir)
        assert isinstance(sub_tileset, TileSet)

        sub_root_bounding_box = sub_tileset.root_tile.bounding_volume
        assert isinstance(sub_root_bounding_box, BoundingVolumeBox)
        sub_root_bounding_box.transform(sub_tileset.root_tile.transform)

        child_bounding_box = tile_child.bounding_volume
        assert isinstance(child_bounding_box, BoundingVolumeBox)
        child_bounding_box.transform(
            merged_tileset.root_tile.transform @ tile_child.transform
        )

        assert_array_almost_equal(sub_root_bounding_box._box, child_bounding_box._box)  # type: ignore [arg-type]

    assert merged_tileset.root_tile.content_uri == Path("preview.pnts")

    merged_pnts = merged_tileset.root_tile.tile_content
    assert isinstance(merged_pnts, Pnts)

    assert merged_pnts.body.feature_table.nb_points() == 1582


def test_merge_with_file_tilesets(
    tmp_dir: Path, tileset_ifc_path_1: Path, tileset_ifc_path_2: Path
) -> None:
    tileset_1 = TileSet.from_file(tileset_ifc_path_1)
    tileset_2 = TileSet.from_file(tileset_ifc_path_2)

    merged_tileset_path = tmp_dir / "merged_tileset.json"

    merge_from_files(
        [tileset_ifc_path_1, tileset_ifc_path_2],
        merged_tileset_path,
        overwrite=False,
    )

    # The main tilesets should not be modified (for the moment)
    assert tileset_1.to_dict() == TileSet.from_file(tileset_ifc_path_1).to_dict()
    assert tileset_2.to_dict() == TileSet.from_file(tileset_ifc_path_2).to_dict()

    merged_tileset = TileSet.from_file(merged_tileset_path)

    assert len(merged_tileset.root_tile.get_all_children()) == 2

    # check if URIs are present
    for result, expected in zip(
        merged_tileset.root_tile.get_all_children(),
        (tileset_ifc_path_1, tileset_ifc_path_2),
    ):
        assert result.content_uri == expected.relative_to(merged_tileset_path.parent)

    assert merged_tileset.geometric_error == max(
        tileset_1.root_tile.geometric_error, tileset_2.root_tile.geometric_error
    )
    assert merged_tileset.root_tile.geometric_error == max(
        tileset_1.root_tile.geometric_error, tileset_2.root_tile.geometric_error
    )

    # check if the bounding boxes are consistent
    for tile_child in merged_tileset.root_tile.get_all_children():
        sub_tileset = tile_child.get_or_fetch_content(tmp_dir)
        assert isinstance(sub_tileset, TileSet)

        sub_root_bounding_box = sub_tileset.root_tile.bounding_volume
        assert isinstance(sub_root_bounding_box, BoundingVolumeBox)
        sub_root_bounding_box.transform(sub_tileset.root_tile.transform)

        child_bounding_box = tile_child.bounding_volume
        assert isinstance(child_bounding_box, BoundingVolumeBox)
        child_bounding_box.transform(
            merged_tileset.root_tile.transform @ tile_child.transform
        )

        assert_array_equal(sub_root_bounding_box._box, child_bounding_box._box)  # type: ignore [arg-type]

    assert merged_tileset.root_tile.content_uri is None
    assert merged_tileset.root_tile.tile_content is None

    merged_tileset_path.unlink()


def test_merge_with_pnts_content_with_file_tilesets(
    tmp_dir: Path, tileset_pnts_path_1: Path, tileset_pnts_path_2: Path
) -> None:
    tileset_1 = TileSet.from_file(tileset_pnts_path_1)
    tileset_2 = TileSet.from_file(tileset_pnts_path_2)

    merged_tileset_path = tmp_dir / "merged_tileset.json"

    merge_from_files(
        [tileset_pnts_path_1, tileset_pnts_path_2],
        merged_tileset_path,
        overwrite=False,
    )

    # The main tilesets should not be modified (for the moment)
    assert tileset_1.to_dict() == TileSet.from_file(tileset_pnts_path_1).to_dict()
    assert tileset_2.to_dict() == TileSet.from_file(tileset_pnts_path_2).to_dict()

    merged_tileset = TileSet.from_file(merged_tileset_path)

    assert len(merged_tileset.root_tile.get_all_children()) == 2

    # check if URIs are present
    for result, expected in zip(
        merged_tileset.root_tile.get_all_children(),
        (tileset_pnts_path_1, tileset_pnts_path_2),
    ):
        assert result.content_uri == expected.relative_to(merged_tileset_path.parent)

    assert merged_tileset.geometric_error == 1720.2520618394917
    assert merged_tileset.root_tile.geometric_error == 1720.2520618394917

    # check if the bounding boxes are consistent
    for tile_child in merged_tileset.root_tile.get_all_children():
        sub_tileset = tile_child.get_or_fetch_content(tmp_dir)
        assert isinstance(sub_tileset, TileSet)

        sub_root_bounding_box = sub_tileset.root_tile.bounding_volume
        assert isinstance(sub_root_bounding_box, BoundingVolumeBox)
        sub_root_bounding_box.transform(sub_tileset.root_tile.transform)

        child_bounding_box = tile_child.bounding_volume
        assert isinstance(child_bounding_box, BoundingVolumeBox)
        child_bounding_box.transform(
            merged_tileset.root_tile.transform @ tile_child.transform
        )

        assert_array_almost_equal(sub_root_bounding_box._box, child_bounding_box._box)  # type: ignore [arg-type]

    merged_pnts_path = merged_tileset_path.parent / "preview.pnts"
    assert merged_pnts_path.exists()

    merged_pnts = read_binary_tile_content(merged_pnts_path)
    assert isinstance(merged_pnts, Pnts)

    assert merged_pnts.body.feature_table.nb_points() == 1582
    assert len(merged_pnts.body.batch_table.get_binary_property("intensity")) == 1582
    assert (
        len(merged_pnts.body.batch_table.get_binary_property("raw_classification"))
        == 1582
    )
