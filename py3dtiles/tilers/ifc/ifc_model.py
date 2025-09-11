import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pygltflib
from pyproj import Transformer

from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox


@dataclass
class FilenameAndOffset:
    filename: str
    offset: list[float]


@dataclass
class FileMetadata:
    offset: list[float] | None
    crs_in: str | None
    transformer: Transformer | None


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
    parent_id: int | None
    transform: npt.NDArray[np.float64] | None
    box: BoundingVolumeBox | None
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
        pbr_metallic_roughness = pygltflib.PbrMetallicRoughness(
            baseColorFactor=base_color_factor, roughnessFactor=0.5, metallicFactor=0.5
        )
        alpha_mode = pygltflib.BLEND if self.transparency else pygltflib.OPAQUE
        return pygltflib.Material(
            pbrMetallicRoughness=pbr_metallic_roughness, alphaMode=alpha_mode
        )


@dataclass
class Geometry:
    """
    A collection of vertices and associated faces.
    """

    verts: list[float]
    """
    List of vertices
    """
    faces: list[int]
    """
    List of face ids, to be matched with material ids
    """

    def compute_bounding_volume_box(self) -> BoundingVolumeBox:
        """
        compute the bbox of this
        """
        vertices_view = np.array(self.verts).reshape((-1, 3))
        maxes = vertices_view.max(axis=0)
        mins = vertices_view.min(axis=0)
        bbox = BoundingVolumeBox()
        bbox.set_from_mins_maxs(np.concatenate([mins, maxes]))
        return bbox


@dataclass
class IfcMesh:
    geom: Geometry
    materials: list[IfcMaterial]
    material_ids: tuple[int] | None


@dataclass
class Feature:
    mesh: IfcMesh | None
    properties: dict[str, Any]


@dataclass
class IfcTile:
    """
    Full ifc tile, containing all the features
    """

    tile_id: int
    filename: Path  # we need to keep the filename to get the information about metadata
    parent_id: int | None
    members: list[Feature]
    properties: dict[Any, Any]
