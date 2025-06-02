from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from py3dtiles.typing import (
    BoundingVolumeBoxDictType,
    BoundingVolumeRegionDictType,
    BoundingVolumeSphereDictType,
)

from .root_property import RootProperty

if TYPE_CHECKING:
    from typing_extensions import Self


_BoundingVolumeJsonDictT = TypeVar(
    "_BoundingVolumeJsonDictT",
    BoundingVolumeBoxDictType,
    BoundingVolumeRegionDictType,
    BoundingVolumeSphereDictType,
)


class BoundingVolume(
    RootProperty[_BoundingVolumeJsonDictT], Generic[_BoundingVolumeJsonDictT]
):
    """
    Abstract class used as interface for box, region and sphere.
    """

    def __init__(self) -> None:
        super().__init__()

    # the following method should probably not be abstract and should construct the correct instance
    # considering the dict format
    @classmethod
    @abstractmethod
    def from_dict(cls, bounding_volume_dict: _BoundingVolumeJsonDictT) -> Self:
        """
        Construct a bounding volume from a dict. The implementor should support each possible format in the 3dtiles spec relevant to the specific flavor of bounding volume.
        """
        ...

    @abstractmethod
    def get_center(self) -> npt.NDArray[np.float64]:
        """
        Returns the center of this bounding volume.
        """
        ...

    @abstractmethod
    def translate(self, offset: npt.NDArray[np.float64]) -> None:
        """
        Translate the volume center with the given offset "vector"

        :param offset: the 3D vector by which the volume should be translated
        """
        ...

    @abstractmethod
    def transform(self, transform: npt.NDArray[np.float64]) -> None:
        """
        Apply the provided transformation matrix (4x4) to the volume

        :param transform: transformation matrix (4x4) to be applied
        """
        ...

    @abstractmethod
    def add(self, other: BoundingVolume[Any]) -> None:
        """
        Compute the 'canonical' bounding volume fitting this bounding volume
        together with the added bounding volume. The computed fitting bounding volume is generically
        not the smallest one (due to its alignment with the coordinate axis).

        :param other: another bounding volume to be added with this one
        """
        ...

    @abstractmethod
    def is_valid(self) -> bool:
        """
        Test if the bounding volume is correctly defined.
        """
        ...

    @abstractmethod
    def get_half_size(self) -> npt.NDArray[np.float64]:
        """
        Returns the half size of this volume as a 3 elements array
        """
        ...

    @abstractmethod
    def to_dict(self) -> _BoundingVolumeJsonDictT:
        """
        Serialize this bounding volume into its JSON representation, ready to be embedded in a tileset.
        """
        ...
