from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from py3dtiles.tileset.bounding_volume_box import BoundingVolumeBox

from .batch_table import BatchTable
from .feature_table import FeatureTable


class TileContent(ABC):

    @abstractmethod
    def to_array(self) -> npt.NDArray[np.uint8]:
        """
        Return a uint8 view of the bytes composing this tile content.

        This is mainly a legacy method used to load from binary files.
        """

    @abstractmethod
    def save_as(self, path: Path) -> None:
        """
        Save the tile content to a file.
        """

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> Self:
        """
        Load a tile_content from bytes. This method can be directly supplied the bytes in the file.
        """

    @classmethod
    def from_file(cls, tile_path: Path) -> Self:
        with tile_path.open("rb") as f:
            data = f.read()
            return cls.from_bytes(data)

    @abstractmethod
    def get_vertex_count(self) -> int: ...

    @abstractmethod
    def get_vertices(self) -> npt.NDArray[np.float32 | np.uint16] | None: ...

    @abstractmethod
    def get_colors(self) -> npt.NDArray[np.uint8 | np.uint16 | np.float32] | None: ...

    @abstractmethod
    def get_extra_field(self, fieldname: str) -> npt.NDArray[Any] | None: ...

    @abstractmethod
    def get_extra_field_names(self) -> set[str]: ...

    def get_bounding_volume_box(self) -> BoundingVolumeBox | None:
        vertices = self.get_vertices()
        if vertices is None:
            return None
        return BoundingVolumeBox.from_points(vertices)


class LegacyTileContent(TileContent):
    header: TileContentHeader
    body: TileContentBody

    @abstractmethod
    def sync(self) -> None:
        """
        Allow to synchronize headers with contents.
        """

    def to_array(self) -> npt.NDArray[np.uint8]:
        self.sync()
        header_arr = self.header.to_array()
        body_arr = self.body.to_array()
        return np.concatenate((header_arr, body_arr))

    @classmethod
    @abstractmethod
    def from_array(cls, array: npt.NDArray[np.uint8]) -> Self: ...

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cls.from_array(arr)

    def save_as(self, path: Path) -> None:
        tile_arr = self.to_array()
        with path.open("bw") as f:
            f.write(bytes(tile_arr))

    def __str__(self) -> str:
        return (
            "------ Tile header ------\n"
            + (str(self.header) if self.header is not None else "")
            + "\n"
            + "------ Tile body ------\n"
            + (str(self.body) if self.body is not None else "")
        )

    def get_extra_field_names(self) -> set[str]:
        return set(self.body.batch_table.header.data.keys())

    def get_batch_table_binary_property(self, name: str) -> npt.NDArray[Any]:
        """
        Get a binary property from a batch table. Internally forward to `self.body.batch_table.get_binary_property(name)`.
        """
        return self.body.batch_table.get_binary_property(name)


class TileContentHeader(ABC):
    magic_value: Literal[b"b3dm", b"pnts"]
    version: int

    def __init__(self) -> None:
        self.tile_byte_length = 0
        self.ft_json_byte_length = 0
        self.ft_bin_byte_length = 0
        self.bt_json_byte_length = 0
        self.bt_bin_byte_length = 0
        self.bt_length = 0  # number of models in the batch

    def __str__(self) -> str:
        infos = {
            "magic": self.magic_value,
            "version": self.version,
            "tile_byte_length": self.tile_byte_length,
            "json_feature_table_length": self.ft_json_byte_length,
            "bin_feature_table_length": self.ft_bin_byte_length,
            "json_batch_table_length": self.bt_json_byte_length,
            "bin_batch_table_length": self.bt_bin_byte_length,
        }
        return "\n".join(f"{key}: {value}" for key, value in infos.items())

    @staticmethod
    @abstractmethod
    def from_array(array: npt.NDArray[np.uint8]) -> TileContentHeader: ...

    @abstractmethod
    def to_array(self) -> npt.NDArray[np.uint8]: ...


class TileContentBody(ABC):
    batch_table: BatchTable
    feature_table: FeatureTable[Any, Any]

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def to_array(self) -> npt.NDArray[np.uint8]: ...
