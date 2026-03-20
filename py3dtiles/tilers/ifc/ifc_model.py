import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pygltflib
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
class Color:
    r: float
    g: float
    b: float

    def to_list(self) -> list[float]:
        return [self.r, self.g, self.b]


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
class Primitive:
    """
    List of face ids, to be matched with material ids
    """

    faces: npt.NDArray[np.uint8 | np.uint16 | np.uint32] | None
    material: IfcMaterial | None

    def to_gltflib_primitive(self) -> GltfPrimitive:
        if self.material is None:
            return GltfPrimitive(triangles=self.faces, material=None)
        else:
            return GltfPrimitive(
                triangles=self.faces, material=self.material.to_pygltflib_material()
            )


@dataclass
class Mesh:
    """
    A mesh, represented by vertices and primitives
    """

    vertices: npt.NDArray[np.float64]
    primitives: list[Primitive]

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
