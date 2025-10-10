from __future__ import annotations

import struct
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import InvalidPntsError
from py3dtiles.points import Points

from .batch_table import BatchTable
from .pnts_feature_table import (
    PntsFeatureTable,
    PntsFeatureTableBody,
    PntsFeatureTableHeader,
    SemanticPoint,
)
from .tile_content import TileContent, TileContentBody, TileContentHeader


def get_color_semantic(
    colors: npt.NDArray[np.uint8 | np.uint16] | None,
) -> Literal[SemanticPoint.RGB, SemanticPoint.RGB565] | None:
    if colors is None:
        return None
    elif colors.dtype.type is np.uint8:
        return SemanticPoint.RGB
    elif colors.dtype.type is np.uint16:
        return SemanticPoint.RGB565
    else:
        raise ValueError(f"dtype {colors.dtype} not supported for colors")


class Pnts(TileContent):
    def __init__(self, header: PntsHeader, body: PntsBody) -> None:
        super().__init__()
        self.header: PntsHeader = header
        self.body: PntsBody = body
        self.sync()

    def sync(self) -> None:
        """
        Synchronizes headers with the Pnts body.
        """
        self.header.ft_json_byte_length = len(self.body.feature_table.header.to_array())
        self.header.ft_bin_byte_length = len(self.body.feature_table.body.to_array())
        self.header.bt_json_byte_length = len(self.body.batch_table.header.to_array())
        self.header.bt_bin_byte_length = len(self.body.batch_table.body.to_array())

        self.header.tile_byte_length = (
            PntsHeader.BYTE_LENGTH
            + self.header.ft_json_byte_length
            + self.header.ft_bin_byte_length
            + self.header.bt_json_byte_length
            + self.header.bt_bin_byte_length
        )

    def get_points(self, transform: npt.NDArray[np.float64] | None) -> Points:
        """
        Get the points optionally transformed by `transform`.

        Internally forward to `self.body.get_points`
        """
        return self.body.get_points(transform)

    @staticmethod
    def from_features(
        feature_table_header: PntsFeatureTableHeader,
        position_array: npt.NDArray[np.float32 | np.uint16],
        color_array: npt.NDArray[np.uint8 | np.uint16] | None = None,
        normal_position: npt.NDArray[np.float32 | np.uint8] | None = None,
    ) -> Pnts:
        """
        Creates a Pnts from features defined by pd_type and cd_type.
        """
        pnts_body = PntsBody()
        pnts_body.feature_table = PntsFeatureTable.from_features(
            feature_table_header, position_array, color_array, normal_position
        )

        pnts = Pnts(PntsHeader(), pnts_body)
        pnts.sync()

        return pnts

    @staticmethod
    def from_array(array: npt.NDArray[np.uint8]) -> Pnts:
        """
        Creates a Pnts from an array
        """

        # build tile header
        h_arr = array[0 : PntsHeader.BYTE_LENGTH]
        pnts_header = PntsHeader.from_array(h_arr)

        if pnts_header.tile_byte_length != len(array):
            raise InvalidPntsError(
                f"Invalid byte length in header, the size of array is {len(array)}, "
                f"the tile_byte_length for header is {pnts_header.tile_byte_length}"
            )

        # build tile body
        b_len = (
            pnts_header.ft_json_byte_length
            + pnts_header.ft_bin_byte_length
            + pnts_header.bt_json_byte_length
            + pnts_header.bt_bin_byte_length
        )
        b_arr = array[PntsHeader.BYTE_LENGTH : PntsHeader.BYTE_LENGTH + b_len]
        pnts_body = PntsBody.from_array(pnts_header, b_arr)

        # build the tile with header and body
        return Pnts(pnts_header, pnts_body)

    @staticmethod
    def from_points(
        points: Points,
    ) -> Pnts:
        """
        Create a pnts from data array:

        - positions will be included as SemanticPoint.POSITION
        - if rgb is not None, it will be included as SemanticPoint.RGB
        - all the extra_fields are included in the batch table

        :param positions: the positions 1D array
        :param colors: the colors 1D array
        :param extra_fields: a dict of extra arrays to include in the batch_table. The dict keys are the name of the fields, the values are 1D np arrays containing values for each field.
        """
        positions = points.positions.ravel()
        count = len(positions) // 3

        colors = None
        if points.colors is not None:
            colors = points.colors.ravel()
            if colors.ndim != 1:
                raise ValueError(
                    f"The 'colors' array should be flatten, got a NDArray with {colors.ndim} dimensions"
                )
            if (len(colors) // 3) != count:
                raise ValueError(
                    "The 'colors' array does not have the same number of elements as the 'position' array"
                )

        ft = PntsFeatureTable()
        color_semantic = get_color_semantic(points.colors)
        ft.header = PntsFeatureTableHeader.from_semantic(
            SemanticPoint.POSITION,
            color_semantic,
            None,
            count,
        )
        ft.body = PntsFeatureTableBody(positions=positions, color=colors)

        bt = BatchTable()
        for fieldname, array in points.extra_fields.items():
            if array.ndim != 1:
                raise ValueError(
                    f"The '{fieldname}' array in 'extra_fields' should be flat, got a NDArray with {array.ndim} dimensions"
                )

            bt.add_property_as_binary(fieldname, array, "SCALAR")

        body = PntsBody()
        body.feature_table = ft
        body.batch_table = bt

        pnts = Pnts(PntsHeader(), body)
        pnts.sync()

        return pnts

    @staticmethod
    def from_file(tile_path: Path) -> Pnts:
        with tile_path.open("rb") as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            return Pnts.from_array(arr)


class PntsHeader(TileContentHeader):
    BYTE_LENGTH = 28

    def __init__(self) -> None:
        super().__init__()
        self.magic_value = b"pnts"
        self.version = 1

    def to_array(self) -> npt.NDArray[np.uint8]:
        """
        Returns the header as a numpy array.
        """
        header_arr = np.frombuffer(self.magic_value, np.uint8)

        header_arr2 = np.array(
            [
                self.version,
                self.tile_byte_length,
                self.ft_json_byte_length,
                self.ft_bin_byte_length,
                self.bt_json_byte_length,
                self.bt_bin_byte_length,
            ],
            dtype=np.uint32,
        )

        return np.concatenate((header_arr, header_arr2.view(np.uint8)))

    @staticmethod
    def from_array(array: npt.NDArray[np.uint8]) -> PntsHeader:
        """
        Create a PntsHeader from an array
        """

        h = PntsHeader()

        if len(array) != PntsHeader.BYTE_LENGTH:
            raise InvalidPntsError(
                f"Invalid header byte length, the size of array is {len(array)}, "
                f"the header must have a size of {PntsHeader.BYTE_LENGTH}"
            )

        h.version = struct.unpack("i", array[4:8].tobytes())[0]
        h.tile_byte_length = struct.unpack("i", array[8:12].tobytes())[0]
        h.ft_json_byte_length = struct.unpack("i", array[12:16].tobytes())[0]
        h.ft_bin_byte_length = struct.unpack("i", array[16:20].tobytes())[0]
        h.bt_json_byte_length = struct.unpack("i", array[20:24].tobytes())[0]
        h.bt_bin_byte_length = struct.unpack("i", array[24:28].tobytes())[0]

        return h


class PntsBody(TileContentBody):
    def __init__(self) -> None:
        self.feature_table: PntsFeatureTable = PntsFeatureTable()
        self.batch_table = BatchTable()

    def __str__(self) -> str:
        infos = {
            "feature_table_header": self.feature_table.header.to_json(),
            "points_length": self.feature_table.header.points_length,
        }
        if self.feature_table.header.points_length > 0:
            (
                feature_position,
                feature_color,
                feature_normal,
            ) = self.feature_table.get_feature_at(0)
            infos["first_point_position"] = feature_position
            infos["first_point_color"] = feature_color
            infos["first_point_normal"] = feature_normal
        infos["batch_table_header"] = self.batch_table.header.data
        for f in self.batch_table.header.data.keys():
            infos[f"- first point {f}"] = self.batch_table.get_binary_property(f)[0]
        return "\n".join(f"{key}: {value}" for key, value in infos.items())

    def to_array(self) -> npt.NDArray[np.uint8]:
        """
        Returns the body as a numpy array.
        """
        feature_table_array = self.feature_table.to_array()
        batch_table_array = self.batch_table.to_array()
        return np.concatenate((feature_table_array, batch_table_array))

    def get_points(self, transform: npt.NDArray[np.float64] | None) -> Points:
        """
        Get the points inside this instance, optionally transformed by `transform`.
        """
        fth = self.feature_table.header

        xyz = self.feature_table.body.position.view(np.float32).reshape(
            (fth.points_length, 3)
        )
        if fth.colors == SemanticPoint.RGB:
            rgb = self.feature_table.body.color
            if rgb is None:
                raise InvalidPntsError(
                    "If fth.colors is SemanticPoint.RGB, rgb cannot be None."
                )
            rgb = rgb.reshape((fth.points_length, 3))
        else:
            rgb = None

        # data in batch table
        extra_fields = {}
        for field in self.batch_table.get_property_names():
            extra_fields[field] = self.batch_table.get_binary_property(field)

        points = Points(positions=xyz, colors=rgb, extra_fields=extra_fields)
        if transform is not None:
            points.transform(transform)

        return points

    @staticmethod
    def from_array(header: PntsHeader, array: npt.NDArray[np.uint8]) -> PntsBody:
        """
        Creates a PntsBody from an array and the header
        """

        # build feature table
        feature_table_size = header.ft_json_byte_length + header.ft_bin_byte_length
        feature_table_array = array[:feature_table_size]
        feature_table = PntsFeatureTable.from_array(header, feature_table_array)

        # build batch table
        batch_table_size = header.bt_json_byte_length + header.bt_bin_byte_length
        batch_table_array = array[
            feature_table_size : feature_table_size + batch_table_size
        ]
        batch_table = BatchTable.from_array(
            header, batch_table_array, feature_table.nb_points()
        )

        # build tile body with feature table
        body = PntsBody()
        body.feature_table = feature_table
        body.batch_table = batch_table

        return body
