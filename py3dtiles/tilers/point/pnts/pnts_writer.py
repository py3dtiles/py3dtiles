from __future__ import annotations

import pickle
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lz4.frame as gzip
import numpy as np
import numpy.typing as npt

import py3dtiles
from py3dtiles.tileset.content import Pnts
from py3dtiles.utils import node_name_to_path

if TYPE_CHECKING:
    from py3dtiles.tilers.point.node import DummyNode, Node
    from py3dtiles.tilers.point.node.node import DummyNodeDictType


def points_to_pnts_file(
    out_folder: Path,
    name: bytes,
    positions: npt.NDArray[np.float32 | np.uint16],
    colors: npt.NDArray[np.uint8 | np.uint16] | None = None,
    extra_fields: dict[str, npt.NDArray[Any]] | None = None,
) -> tuple[int, Path]:
    """
    Write a pnts file from an uint8 data array containing:
     - points as SemanticPoint.POSITION
     - if include_rgb, rgb as SemanticPoint.RGB
     - if include_classification, classification as a single np.uint8 value
     - if include_intensity, intensity as a single np.uint8 value
    """
    pnts = Pnts.from_points(positions, colors, extra_fields)

    node_path = node_name_to_path(out_folder, name, ".pnts")

    if node_path.exists():
        raise FileExistsError(f"{node_path} already written")

    pnts.save_as(node_path)

    return pnts.body.feature_table.nb_points(), node_path


def node_to_pnts(
    name: bytes,
    node: Node | DummyNode,
    out_folder: Path,
) -> int:
    points = node.get_points()

    if points is None:
        return 0
    point_nb, _ = points_to_pnts_file(
        out_folder,
        name,
        points.positions,
        colors=points.colors,
        extra_fields=points.extra_fields,
    )
    return point_nb


def run(
    data: bytes,
    folder: Path,
) -> Generator[int, None, None]:
    # we can safely write the .pnts file
    if len(data) > 0:
        root = pickle.loads(gzip.decompress(data))
        total = 0
        for name in root:
            node_data: DummyNodeDictType = pickle.loads(root[name])
            node = py3dtiles.tilers.point.node.DummyNode(node_data)
            total += node_to_pnts(name, node, folder)
        yield total
