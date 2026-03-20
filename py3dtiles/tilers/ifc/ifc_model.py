from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox
from py3dtiles.tileset.content.gltf_utils import GltfPrimitive


@dataclass
class FilenameAndOffset:
    filename: str
    offset: npt.NDArray[np.float64]


@dataclass
class FileMetadata:
    offset: npt.NDArray[np.float64] | None
    crs_in: str | None
    transformer: Transformer | None


@dataclass
class Mesh:
    """
    A mesh, represented by vertices and primitives
    """

    vertices: npt.NDArray[np.float64]
    primitives: list[GltfPrimitive]

    def compute_bounding_volume_box(self) -> BoundingVolumeBox:
        """
        compute the bbox of this
        """
        vertices_view = np.array(self.vertices).reshape((-1, 3))
        maxes = vertices_view.max(axis=0)
        mins = vertices_view.min(axis=0)
        bbox = BoundingVolumeBox()
        bbox.set_from_mins_maxs(np.concatenate([mins, maxes]))
        return bbox


@dataclass
class Feature:
    mesh: Mesh | None
    properties: dict[str, Any]


@dataclass
class FeatureGroup:
    """
    A data structure grouping some features, with properties. Usually, this is the first representation of a Tile on memory just after reading feature from a dataset.
    """

    tile_id: int
    filename: Path  # we need to keep the filename to get the information about metadata
    parent_id: int | None
    members: list[Feature]
    properties: dict[Any, Any]


@dataclass
class TileInfo:
    """
    Lightweight ifc tile info used to keep tile metadata in-memory after a tile has been written to disk.

    It encapsulates all the necessary informations to be able to reconstruct the tileset hierarchy from a set of tiles.
    """

    tile_id: int
    parent_id: int | None
    transform: npt.NDArray[np.float64] | None
    box: BoundingVolumeBox | None
    elem_max_size: float
    has_content: bool
    properties: dict[str, str]
