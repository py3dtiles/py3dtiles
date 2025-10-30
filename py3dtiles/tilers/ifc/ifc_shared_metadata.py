from dataclasses import dataclass
from pathlib import Path

from py3dtiles.tilers.base_tiler import SharedMetadata


@dataclass(frozen=True)
class IfcSharedMetadata(SharedMetadata):
    verbosity: int
    out_folder: Path
