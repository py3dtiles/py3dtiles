from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from py3dtiles.tilers.base_tiler import SharedMetadata
from py3dtiles.typing import ExtraFieldsDescription


@dataclass(frozen=True)
class PointSharedMetadata(SharedMetadata):
    transformer: Transformer | None
    root_aabb: npt.NDArray[np.float64]
    root_spacing: float
    scale: npt.NDArray[np.float32]
    out_folder: Path
    write_rgb: bool
    color_scale: float | None
    extra_fields_to_include: list[ExtraFieldsDescription]
    verbosity: int
