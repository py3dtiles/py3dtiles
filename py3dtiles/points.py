from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from numpy import typing as npt


@dataclass
class Points:
    """
    This class represents an arbitrary amount of points.

    Each properties contains one elem per *vertices*. Ex: positions is an array of coordinates triplet.
    """

    positions: npt.NDArray[Union[np.float32, np.uint16]]
    # colors is an array of RGB triplet
    colors: Optional[npt.NDArray[Union[np.uint8, np.uint16]]]
    # etc...
    extra_fields: dict[str, npt.NDArray[Any]]


@dataclass
class FlatPoints:
    """
    This class represents an arbitrary amount of points.

    Each properties is a flat array containing every value.

    Positions and rgb have an implicit stride of 3, while extra_fields contains scalar values in each dict value. At the moment, we only support scalar values for extra fields.
    """

    # implicit stride of 3
    positions: npt.NDArray[Union[np.float32, np.uint16]]
    colors: Optional[npt.NDArray[Union[np.uint8, np.uint16]]]
    extra_fields: dict[str, npt.NDArray[Any]]
