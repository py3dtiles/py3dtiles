from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from py3dtiles.exceptions import Invalid3dtilesError

from .b3dm import B3dm
from .gltf import Gltf, PointsGltf
from .pnts import Pnts

if TYPE_CHECKING:
    from py3dtiles.tileset.content import TileContent

__all__ = ["read_binary_tile_content"]


def read_binary_tile_content(tile_path: Path) -> TileContent:
    with tile_path.open("rb") as f:
        data = f.read()
        magic = data[:4]
        tile_content: TileContent | None = None
        if magic == b"pnts":
            tile_content = Pnts.from_bytes(data)
        if magic == b"b3dm":
            tile_content = B3dm.from_bytes(data)
        if magic == b"glTF":
            tile_content = Gltf.from_bytes(data)
            if PointsGltf.can_build_from_gltf(tile_content._gltf):
                tile_content = PointsGltf(tile_content._gltf)

        if tile_content is None:
            raise Invalid3dtilesError(
                f"The file {tile_path} doesn't contain a valid TileContent data."
            )

        return tile_content
