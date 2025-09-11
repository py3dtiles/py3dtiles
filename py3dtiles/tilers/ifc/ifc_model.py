import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pygltflib
from pyproj import Transformer

from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox


@dataclass
class FileMetadata:
    offset: Optional[list[float]]
    crs_in: Optional[str]
    transformer: Optional[Transformer]


@dataclass
class Color:
    r: float
    g: float
    b: float

    def to_list(self) -> list[float]:
        return [self.r, self.g, self.b]


@dataclass
class IfcTileInfo:
    """
    Lightweight ifc tile info, to be able to reconstruct the tileset
    """

    tile_id: int
    parent_id: Optional[int]
    transform: Optional[npt.NDArray[np.float64]]
    box: Optional[BoundingVolumeBox]
    elem_max_size: float
    has_content: bool
    properties: dict[str, str]


@dataclass
class IfcMaterial:
    diffuse: Color
    specular: Color
    specularity: float
    transparency: float

    def to_pygltflib_material(self) -> pygltflib.Material:
        base_color_factor = self.diffuse.to_list()
        transparency = (
            self.transparency
            if self.transparency and not math.isnan(self.transparency)
            else 0.0
        )
        base_color_factor.append(1.0 - transparency)
        # roughnessFactor = (
        #     1.0 / self.specularity
        #     if self.specularity and not math.isnan(self.specularity)
        #     else None
        # )
        pbrMetallicRoughness = pygltflib.PbrMetallicRoughness(
            baseColorFactor=base_color_factor, roughnessFactor=0.5, metallicFactor=0.5
        )
        alpha_mode = pygltflib.BLEND if self.transparency else pygltflib.OPAQUE
        return pygltflib.Material(
            pbrMetallicRoughness=pbrMetallicRoughness, alphaMode=alpha_mode
        )


@dataclass
class Geometry:
    verts: list[float]
    """
    List of face ids, to be matched with material ids
    """
    faces: list[int]

    def compute_bounding_volume_box(self) -> BoundingVolumeBox:
        """
        compute the bbox of this
        """
        vertices_view = np.array(self.verts).reshape((-1, 3))
        maxes = vertices_view.max(0)
        mins = vertices_view.min(0)
        bbox = BoundingVolumeBox()
        bbox.set_from_mins_maxs(np.concatenate([mins, maxes]))
        return bbox


@dataclass
class IfcMesh:
    geom: Geometry
    materials: list[IfcMaterial]
    material_ids: Optional[tuple[int]]


@dataclass
class Feature:
    mesh: Optional[IfcMesh]
    properties: dict[str, Any]


@dataclass
class IfcTile:
    """
    Full ifc tile, containing all the features
    """

    tile_id: int
    filename: Path  # we need to keep the filename to get the information about metadata
    parent_id: Optional[int]
    members: list[Feature]
    properties: dict[Any, Any]
