from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from numpy import typing as npt


@dataclass
class Points:
    """
    This class represents an arbitrary amount of points.

    Each properties contains one elem per *vertices*. Ex: positions is an array of coordinates triplet.

    Ex: [[1, 2, 3], [3, 4, 5]]

    `extra_fields` should be a dict. The property name is the field name, the values is a flat array of values
    """

    positions: npt.NDArray[Union[np.float32, np.uint16]]
    # colors is an array of RGB triplet
    colors: Optional[npt.NDArray[Union[np.uint8, np.uint16]]]
    # etc...
    extra_fields: dict[str, npt.NDArray[Any]]

    def __init__(
        self,
        positions: npt.NDArray[Union[np.float32, np.uint16]],
        colors: Optional[npt.NDArray[Union[np.uint8, np.uint16]]] = None,
        extra_fields: Optional[dict[str, npt.NDArray[Any]]] = None,
    ):
        if positions.ndim != 2:
            raise ValueError("Positions parameter should have 2 dimensions")
        if positions.shape[1] != 3:
            raise ValueError("Positions should be an array of coordinates triplet")
        self.positions = positions

        if colors is None:
            self.colors = None
        else:
            if colors.ndim != 2:
                raise ValueError("colors parameter should have 2 dimensions")
            if colors.shape[1] != 3:
                raise ValueError("colors should be an array of coordinates triplet")
            self.colors = colors

        self.extra_fields = {} if extra_fields is None else extra_fields
