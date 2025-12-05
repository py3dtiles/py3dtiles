from __future__ import annotations

from typing import Any, TypeVar, Union

import numpy as np
import numpy.typing as npt
from numba import njit

from py3dtiles.exceptions import TilerException
from py3dtiles.points import Points
from py3dtiles.typing import ExtraFieldsDescription
from py3dtiles.utils import SubdivisionType, aabb_size_to_subdivision_type

from .distance import is_point_far_enough, xyz_to_key

T = TypeVar(
    "T",
    bound=Union[
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    ],
)


@njit(fastmath=True, cache=True)  # type: ignore [untyped-decorator]
def _insert(
    keys: npt.NDArray[np.int32],
    cells_xyz: list[npt.NDArray[np.float32 | np.uint16]],
    cells_rgb: list[npt.NDArray[np.uint8 | np.uint16]] | None,
    cell_count: npt.NDArray[np.int32],
    xyz: npt.NDArray[np.float32 | np.uint16],
    rgb: npt.NDArray[np.uint8 | np.uint16] | None,
    spacing: float,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.uint8] | None,
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    bool,
]:  # pragma: no cover
    inserted = np.full(len(xyz), True)
    notinserted = np.full(len(xyz), False)
    needs_balance = False

    for i in range(len(xyz)):
        k = keys[i]
        if cells_xyz[k].shape[0] == 0 or is_point_far_enough(
            cells_xyz[k], xyz[i], spacing
        ):
            cells_xyz[k] = np.concatenate((cells_xyz[k], xyz[i].reshape(1, 3)))
            if cells_rgb is not None:
                if rgb is None:
                    raise ValueError("rgb cannot be None if cells_rgb is None")
                cells_rgb[k] = np.concatenate((cells_rgb[k], rgb[i].reshape(1, 3)))

            if cell_count[0] < 8:
                needs_balance = needs_balance or cells_xyz[k].shape[0] > 200000
        else:
            inserted[i] = False
            notinserted[i] = True

    return (
        xyz[notinserted],
        None if rgb is None else rgb[notinserted],
        inserted,
        notinserted,
        needs_balance,
    )


@njit(fastmath=True, cache=True)  # type: ignore [untyped-decorator]
def _insert_extra_fields(
    keys: npt.NDArray[np.int32],
    inserted: npt.NDArray[np.bool_],
    notinserted: npt.NDArray[np.bool_],
    cells_extra_field: list[npt.NDArray[T]],
    extra_field: npt.NDArray[T],
) -> npt.NDArray[T]:  # pragma: no cover
    for i in range(len(keys)):
        k = keys[i]
        if inserted[i]:
            cells_extra_field[k] = np.append(cells_extra_field[k], extra_field[i])
    return extra_field[notinserted]  # type: ignore [no-any-return] # for some reason, mypy doesn't understand that


class Grid:
    """docstring for Grid"""

    __slots__ = (
        "cell_count",
        "cells_xyz",
        "cells_rgb",
        "cells_extra_fields",
        "spacing",
    )

    def __init__(
        self,
        spacing: float,
        include_rgb: bool,
        extra_fields: list[ExtraFieldsDescription],
        initial_count: int = 3,
    ) -> None:
        self.cell_count = np.array(
            [initial_count, initial_count, initial_count], dtype=np.int32
        )
        self.spacing = spacing * spacing

        self.cells_xyz: list[npt.NDArray[np.uint16 | np.float32]] = []
        self.cells_rgb: list[npt.NDArray[np.uint8 | np.uint16]] | None = None
        if include_rgb:
            self.cells_rgb = []
        self.cells_extra_fields: dict[str, list[npt.NDArray[Any]]] = {
            f.name: [] for f in extra_fields
        }
        for _ in range(self.max_key_value):
            self.cells_xyz.append(np.zeros((0, 3), dtype=np.float32))
            if self.cells_rgb is not None:
                self.cells_rgb.append(np.zeros((0, 3), dtype=np.uint8))
            for f in extra_fields:
                self.cells_extra_fields[f.name].append(np.zeros(0, dtype=f.dtype))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "cell_count": self.cell_count,
            "spacing": self.spacing,
            "cells_xyz": self.cells_xyz,
            "cells_rgb": self.cells_rgb,
            "cells_extra_fields": self.cells_extra_fields,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.cell_count = state["cell_count"]
        self.spacing = state["spacing"]
        self.cells_xyz = state["cells_xyz"]
        self.cells_rgb = state["cells_rgb"] if "cells_rgb" in state else None
        self.cells_extra_fields = state["cells_extra_fields"]

    @property
    def max_key_value(self) -> int:
        return 1 << (
            2 * int(self.cell_count[0]).bit_length()
            + int(self.cell_count[2]).bit_length()
        )

    def insert(
        self,
        aabmin: npt.NDArray[np.float32],
        inv_aabb_size: npt.NDArray[np.float32],
        xyz: npt.NDArray[np.float32 | np.uint16],
        rgb: npt.NDArray[np.uint8 | np.uint16] | None,
        extra_fields: dict[str, npt.NDArray[T]],
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.uint8] | None,
        dict[
            str,
            (
                npt.NDArray[np.int8]
                | npt.NDArray[np.int16]
                | npt.NDArray[np.int32]
                | npt.NDArray[np.int64]
                | npt.NDArray[np.uint8]
                | npt.NDArray[np.uint16]
                | npt.NDArray[np.uint32]
                | npt.NDArray[np.uint64]
                | npt.NDArray[np.float16]
                | npt.NDArray[np.float32]
                | npt.NDArray[np.float64]
                | npt.NDArray[np.longdouble]
            ),
        ],
        bool,
    ]:
        # We have to separate the insertion of xyz + rgb from the insertion of extra fields because numba isn't able
        # to take such a dict as argument
        # it implies quite a lot of coupling between this method, _insert and _insert_extra_fields unfortunately
        keys = xyz_to_key(xyz, self.cell_count, aabmin, inv_aabb_size)

        (
            not_inserted_xyz,
            not_inserted_rgb,
            inserted_indices,
            not_inserted_indices,
            needs_balance,
        ) = _insert(
            keys,
            self.cells_xyz,
            self.cells_rgb,
            self.cell_count,
            xyz,
            rgb,
            self.spacing,
        )
        not_inserted_extra_fields = {}
        for f in extra_fields:
            not_inserted_extra_fields[f] = _insert_extra_fields(
                keys,
                inserted_indices,
                not_inserted_indices,
                self.cells_extra_fields[f],
                extra_fields[f],
            )
        return (
            not_inserted_xyz,
            not_inserted_rgb,
            not_inserted_extra_fields,
            needs_balance,
        )

    def init_cells(
        self,
        aabmin: npt.NDArray[np.float32],
        inv_aabb_size: npt.NDArray[np.float32],
        xyz: npt.NDArray[np.float32 | np.uint16],
        rgb: npt.NDArray[np.uint8 | np.uint16] | None,
        extra_fields: dict[str, npt.NDArray[Any]],
    ) -> None:
        keys = xyz_to_key(xyz, self.cell_count, aabmin, inv_aabb_size)
        # allocate this one once and for all
        for k in np.unique(keys):
            mask = keys - k == 0
            self.cells_xyz[k] = np.concatenate((self.cells_xyz[k], xyz[mask]))
            if self.cells_rgb is not None:
                if rgb is None:
                    raise ValueError("rgb cannot be None if this grid expects rgb")
                self.cells_rgb[k] = np.concatenate((self.cells_rgb[k], rgb[mask]))
            for f, arr in extra_fields.items():
                if f in self.cells_extra_fields:
                    self.cells_extra_fields[f][k] = np.concatenate(
                        self.cells_extra_fields[f][k], arr[mask]
                    )
                else:
                    self.cells_extra_fields[f] = [arr[mask]]
        return None

    def needs_balance(self) -> bool:
        if self.cell_count[0] < 8:
            for cell in self.cells_xyz:
                if cell.shape[0] > 100000:
                    return True
        return False

    def balance(
        self,
        aabb_size: npt.NDArray[np.float32],
        aabmin: npt.NDArray[np.float32],
        inv_aabb_size: npt.NDArray[np.float32],
    ) -> None:
        t = aabb_size_to_subdivision_type(aabb_size)
        self.cell_count[0] += 1
        self.cell_count[1] += 1
        if t != SubdivisionType.QUADTREE:
            self.cell_count[2] += 1
        if self.cell_count[0] > 8:
            raise TilerException(
                f"The first value of the attribute cell count must be lower or equal to 8, "
                f"currently, it is {self.cell_count[0]}"
            )

        old_cells_xyz = self.cells_xyz
        self.cells_xyz = []
        if self.cells_rgb is not None:
            old_cells_rgb = self.cells_rgb
            self.cells_rgb = []
        # shallow copy
        old_cells_extra_fields = self.cells_extra_fields.copy()

        for f in self.cells_extra_fields:
            self.cells_extra_fields[f] = []
        for _ in range(self.max_key_value):
            self.cells_xyz.append(np.zeros((0, 3), dtype=np.float32))
            if self.cells_rgb:
                self.cells_rgb.append(np.zeros((0, 3), dtype=np.uint8))
            for f, arr in self.cells_extra_fields.items():
                # if we balance, we must have at least one elem right?
                arr.append(np.zeros(0, dtype=old_cells_extra_fields[f][0].dtype))

        for i in range(len(old_cells_xyz)):
            cell_extra = {}
            for f, arr in old_cells_extra_fields.items():
                cell_extra[f] = arr[i]
            self.init_cells(
                aabmin,
                inv_aabb_size,
                old_cells_xyz[i],
                old_cells_rgb[i],
                cell_extra,
            )

    def get_points(self) -> Points | None:
        if len(self.cells_xyz) == 0:
            # we haven't inserted anything yet
            return None
        xyz = np.zeros((self.get_point_count(), 3), self.cells_xyz[0].dtype)
        rgb = None
        if self.cells_rgb is not None:
            rgb = np.zeros((self.get_point_count(), 3), self.cells_rgb[0].dtype)
        extra_fields: dict[str, npt.NDArray[Any]] = {}
        for f in self.cells_extra_fields:
            extra_fields[f] = np.zeros(
                self.get_point_count(), self.cells_extra_fields[f][0].dtype
            )

        idx = 0
        for i in range(len(self.cells_xyz)):
            cell_xyz = self.cells_xyz[i]
            for j in range(len(cell_xyz)):
                xyz[idx] = cell_xyz[j]
                if self.cells_rgb is not None and rgb is not None:
                    rgb[idx] = self.cells_rgb[i][j]
                # We only support SCALAR additional fields
                for f, arr in self.cells_extra_fields.items():
                    extra_fields[f][idx] = arr[i][j]
                idx += 1

        return Points(positions=xyz, colors=rgb, extra_fields=extra_fields)

    def get_point_count(self) -> int:
        pt = 0
        for i in range(len(self.cells_xyz)):
            pt += self.cells_xyz[i].shape[0]
        return pt
