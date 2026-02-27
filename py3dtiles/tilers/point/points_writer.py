from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from py3dtiles.tileset.content import Pnts
from py3dtiles.tileset.content.gltf import PointsGltf
from py3dtiles.utils import node_name_to_path

if TYPE_CHECKING:
    from py3dtiles.tilers.point.node import DummyNode, Node


def node_to_content(
    name: bytes,
    node: Node | DummyNode,
    out_folder: Path,
    legacy_format: bool,
) -> int:
    points = node.get_points()

    if points is None:
        return 0

    cls = Pnts if legacy_format else PointsGltf
    content = cls.from_points(points)

    ext = ".pnts" if legacy_format else ".glb"
    node_path = node_name_to_path(out_folder, name, ext)

    if node_path.exists():
        raise FileExistsError(f"{node_path} already written")
    content.save_as(node_path)

    return content.get_vertex_count()
