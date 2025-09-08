from dataclasses import dataclass
from typing import Any

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

    positions: npt.NDArray[np.float32 | np.uint16]
    # colors is an array of RGB triplet
    colors: npt.NDArray[np.uint8 | np.uint16] | None
    # etc...
    extra_fields: dict[str, npt.NDArray[Any]]

    def __init__(
        self,
        positions: npt.NDArray[np.float32 | np.uint16],
        colors: npt.NDArray[np.uint8 | np.uint16] | None = None,
        extra_fields: dict[str, npt.NDArray[Any]] | None = None,
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

    def transform(self, transform: npt.NDArray[np.float64]) -> None:
        transform = transform.reshape((4, 4), order="F")
        xyz = self.positions
        xyzw = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)))
        xyz = xyzw.dot(transform.T.astype(xyz.dtype))[:, :3]
        self.positions = xyz
