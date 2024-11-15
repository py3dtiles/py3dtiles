from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from py3dtiles.tileset.content import Pnts
from py3dtiles.utils import node_name_to_path

if TYPE_CHECKING:
    from py3dtiles.points import Points
    from py3dtiles.tilers.point.node import DummyNode, Node


def points_to_pnts_file(
    out_folder: Path,
    name: bytes,
    points: Points,
) -> tuple[int, Path]:
    """
    Write a pnts file from an uint8 data array containing:
     - points as SemanticPoint.POSITION
     - if include_rgb, rgb as SemanticPoint.RGB
     - if include_classification, classification as a single np.uint8 value
     - if include_intensity, intensity as a single np.uint8 value
    """
    pnts = Pnts.from_points(points)

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
    point_nb, _ = points_to_pnts_file(out_folder, name, points)
    return point_nb
