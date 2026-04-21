from abc import ABC
from dataclasses import dataclass
from pathlib import Path

from py3dtiles.constants import SpecVersion


@dataclass(frozen=True)
class SharedMetadata(ABC):
    """
    Base class with data that must be shared with worker tiler.
    """

    spec_version: SpecVersion
    verbosity: int
    out_folder: Path
