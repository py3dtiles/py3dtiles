import argparse
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import (
    TileContentMissingException,
)
from py3dtiles.points import Points
from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.content import Pnts
from py3dtiles.tileset.content.tile_content import TileContent
from py3dtiles.tileset.tile import Tile
from py3dtiles.tileset.tileset import TileSet

_T = TypeVar("_T", bound=npt.NBitBase)


_MAX_POINTS_IN_PREVIEW = 50_000


def _get_preview_tile_from_tiles(
    tiles: list[Tile],
    inv_transform: npt.NDArray[np.float64],
) -> tuple[TileContent, float] | None:
    """
    Get a preview of all the tilesets.

    At the moment, it returns a Pnts, but later will return directly a Gltf
    """
    # take half points from our children
    xyz = np.zeros((0, 3), dtype=np.float32)
    rgb = None
    extra_fields: dict[str, npt.NDArray[Any]] = {}

    point_count = 0
    for tile in tiles:

        # but if it is a tileset, let's get the content of the root tile
        if (
            isinstance(tile.tile_content, TileSet)
            and tile.tile_content.root_tile is not None
        ):
            tile = tile.tile_content.root_tile

        if tile.has_content() and not tile.has_content_loaded():
            # at this point we know that content_uri is not None
            # this next line is for the type checker :-)
            assert tile.content_uri is not None
            # we cannot load the tile here, because we need the root_uri from the tileset
            # So we raise it to the caller so that the content loading is done where it can be done
            raise TileContentMissingException(tile.content_uri)

        if isinstance(tile.tile_content, Pnts):
            point_count += tile.tile_content.body.feature_table.header.points_length

    if point_count == 0:
        # at the moment, we support only points, so let's bail out now if there is none
        return None

    ratio = min(1, _MAX_POINTS_IN_PREVIEW / point_count)
    geometric_errors_with_ratio = []

    for tile in tiles:
        tile_content = tile.tile_content
        if isinstance(tile_content, TileSet):
            tile_content = tile_content.root_tile.tile_content
        if not isinstance(tile_content, Pnts):
            # at the moment we can only create previews of Pnts tile
            geometric_errors_with_ratio.append(tile.geometric_error)
            continue

        geometric_errors_with_ratio.append(tile.geometric_error / ratio)

        # get points in absolute coordinates
        points = tile_content.body.get_points(tile.transform)
        _xyz = points.positions
        _rgb = points.colors
        _extra_fields = points.extra_fields

        select = np.random.choice(_xyz.shape[0], int(_xyz.shape[0] * ratio))

        # deal with new fields found in the current tileset
        # note: we have to do this *before* inserting the new positions,
        # so that we can fill with 0 with the correct length more easily
        # if the previous iterations did not contain this field

        # deal with the fields found in previous pnts first
        for field, arr in extra_fields.items():
            if field in _extra_fields:
                extra_fields[field] = np.concatenate(
                    (arr, _extra_fields[field][select])
                )
        # fields new in this pnts
        for field, arr in _extra_fields.items():
            if field not in extra_fields:
                extra_fields[field] = np.concatenate(
                    (
                        np.zeros(len(xyz), dtype=_extra_fields[field].dtype),
                        arr[select],
                    )
                )

        xyz = np.concatenate((xyz, _xyz[select]))
        if _rgb is not None:
            if rgb is None:
                rgb = _rgb[select]
            else:
                rgb = np.concatenate((rgb, _rgb[select]))

    points = Points(positions=xyz, colors=rgb, extra_fields=extra_fields)
    points.transform(inv_transform)

    geometric_error = max(geometric_errors_with_ratio)
    return Pnts.from_points(points), geometric_error


def _get_transform_from_root_tile(root_tile: Tile) -> npt.NDArray[np.float64]:
    assert root_tile.bounding_volume is not None

    # create an offset, the center of the bounding box.
    transform = np.identity(4)
    transform[:3, 3] = root_tile.bounding_volume.get_center()
    return transform


def create_tileset_from_root_tiles(root_tiles: list[Tile]) -> TileSet:
    """
    Creates a TileSet from all the root tiles. This method will calculate a
    reasonable world-transformation and reverse-apply it to each tile
    transformation before adding them to the tileset.

    NOTE: each element of root_tiles *must* have its content loaded if their content uri is defined before calling this method.

    The resulting TileSet might have a root tile with contents. It's the responsibility of the caller to save this content on disk (see :obj:`py3dtiles.tileset.tile.Tile.write_content`).
    """
    global_tileset = TileSet()
    root_tile = global_tileset.root_tile
    # create the hierarchy : one intermediate wrapping tile for each tile
    for tile in root_tiles:
        root_tile.add_child(tile)

    root_tile.set_refine_mode("REPLACE")

    # make sure bounding volume is init
    root_tile.sync_bounding_volume_with_children()
    assert isinstance(root_tile.bounding_volume, BoundingVolumeBox)

    # create an offset, the center of the bounding box.
    transform = _get_transform_from_root_tile(root_tile)
    inv_transform = np.linalg.inv(transform)
    root_tile.transform = transform
    root_tile.bounding_volume.transform(inv_transform)

    # preview tile
    preview = _get_preview_tile_from_tiles(root_tiles, inv_transform)
    if preview is not None:
        root_tile.tile_content, geometric_error = preview
        root_tile.content_uri = Path("./preview.pnts")
    # now combine the inverse transformation to the children tiles
    # this must be done last, so that the preview points are calculated correctly
    for tile in root_tile.children:
        tile.transform = inv_transform @ tile.transform

    # The geometric error
    corners = root_tile.bounding_volume.get_corners()
    geometric_error = float(np.linalg.norm(corners[-1] - corners[0]))
    # geometric error
    global_tileset.geometric_error = geometric_error
    root_tile.geometric_error = geometric_error

    return global_tileset


def merge(
    tilesets: list[TileSet], tileset_paths: dict[TileSet, Path] | None = None
) -> TileSet:
    """
    Create a tileset that include all input tilesets. The tilesets don't need to be written.
    The output tileset is not written but return as a TileSet instance. Please note that the TileSet might have a content in its root tile, not written to disk by this method either.
    """
    if not tilesets:
        raise ValueError("The tileset list cannot be empty")

    tiles = []
    for tileset in tilesets:
        # get a world coordinate bounding volume
        bounding_volume = tileset.root_tile.get_transformed_bounding_volume()
        tile = Tile(
            geometric_error=tileset.root_tile.geometric_error,
            bounding_volume=bounding_volume,
            refine_mode="REPLACE",
        )
        if tileset_paths is not None:
            tile.content_uri = tileset_paths[tileset]
        tile.tile_content = tileset
        tiles.append(tile)

    return create_tileset_from_root_tiles(tiles)


def merge_from_files(
    tileset_paths: list[Path],
    output_tileset_path: Path,
    overwrite: bool = True,
) -> None:
    output_tileset_path = output_tileset_path.absolute()
    if output_tileset_path.exists():
        if overwrite:
            TileSet.from_file(output_tileset_path).delete_on_disk(
                output_tileset_path, delete_sub_tileset=False
            )
        else:
            raise FileExistsError(
                f"Destination tileset {output_tileset_path} already exists."
            )

    tilesets = []
    for path in tileset_paths:
        tileset = TileSet.from_file(path)
        if tileset.root_tile.has_content():
            tileset.root_tile.get_or_fetch_content(tileset.root_uri)
        tilesets.append(tileset)

    relative_tileset_paths = {
        tileset: path.absolute().relative_to(output_tileset_path.parent)
        for tileset, path in zip(tilesets, tileset_paths)
    }

    tileset = merge(tilesets, relative_tileset_paths)

    tileset.root_uri = output_tileset_path.parent
    if tileset.root_tile.has_content():
        tileset.root_tile.write_content(tileset.root_uri)
    tileset.write_as_json(output_tileset_path)


def _init_parser(
    subparser: "argparse._SubParsersAction[Any]",
) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparser.add_parser(
        "merge",
        help="Merge several pointcloud tilesets in 1 tileset. All input tilesets must be relative to the output tileset.",
    )
    parser.add_argument("tilesets", nargs="+", help="All tileset paths to merge")
    parser.add_argument(
        "--output-tileset", required=True, help="The path to the output tileset."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output folder if it already exists.",
    )

    return parser


def _main(args: argparse.Namespace) -> None:
    return merge_from_files(
        [Path(tileset_file) for tileset_file in args.tilesets],
        Path(args.output_tileset),
        args.overwrite,
    )
