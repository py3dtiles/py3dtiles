"""
Reads points from a .xyz or .csv file

Consider XYZIRGB format following FME documentation(*). We do the
following hypothesis and enhancements:

- A header line defining columns in CSV style may be present, but will be ignored (please open an issue if you have a use case where the header is important)
- The separator separating the columns is automagically guessed by the
  reader. This is generally fail safe. It will not harm to use commonly
  accepted separators like space, tab, colon, semi-colon.
- The order of columns is fixed. The reader does the following assumptions:

  - 3 columns mean XYZ
  - 4 columns mean XYZI
  - 6 columns mean XYZRGB
  - 7 columns mean XYZIRGB
  - 8 columns mean XYZIRGB followed by classification data. Classification data must be integers only.
  - all columns after the 8th column will be ignored.

NOTE: we assume RGB are 8 bits components.

(*) See: https://docs.safe.com/fme/html/FME_Desktop_Documentation/FME_ReadersWriters/pointcloudxyz/pointcloudxyz.htm
"""

import csv
import math
from collections.abc import Iterator
from io import TextIOBase
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from py3dtiles.typing import (
    ExtraFieldsDescription,
    MetadataReaderType,
    OffsetScaleType,
    PortionItemType,
)


def get_csv_infos(f: TextIOBase) -> tuple[bool, int, str]:
    file_sample = f.read(2048)  # For performance reasons we just snif the first part
    dialect = csv.Sniffer().sniff(file_sample)
    has_header = csv.Sniffer().has_header(file_sample)
    f.seek(0)
    # skip eventual header
    if has_header:
        f.readline()
    feature_nb = len(f.readline().split(dialect.delimiter))

    f.seek(0)
    return has_header, feature_nb, dialect.delimiter


def get_colors(
    points: npt.NDArray[np.float32],
    feature_nb: int,
    color_scale: float | None,
    with_rgb: bool,
) -> npt.NDArray[np.uint8] | None:
    if with_rgb and feature_nb < 6:
        return np.zeros((len(points), 3), dtype=np.uint8)
    elif with_rgb:
        start = 4 if feature_nb >= 7 else 3
        colors = points[:, start : start + 3]
        if color_scale is not None:
            colors = np.clip(colors * color_scale, 0, 255)
        return colors.astype(np.uint8)
    else:
        return None


def get_metadata(path: Path, color_scale: float | None = None) -> MetadataReaderType:
    aabb = None
    point_count = 0
    seek_values = []

    with path.open() as f:
        has_header, feature_nb, delimiter = get_csv_infos(f)

        if has_header:
            f.readline()

        while True:
            batch = 10_000
            points = np.zeros((batch, 3))

            offset = f.tell()
            for i in range(batch):
                line = f.readline()
                if not line:
                    points = np.resize(points, (i, 3))
                    break
                points[i] = [float(s) for s in line.split(delimiter)][:3]

            if points.shape[0] == 0:
                break

            if not point_count % 1_000_000:
                seek_values += [offset]

            point_count += points.shape[0]
            batch_aabb = np.array([np.min(points, axis=0), np.max(points, axis=0)])

            # Update aabb
            if aabb is None:
                aabb = batch_aabb
            else:
                aabb[0] = np.minimum(aabb[0], batch_aabb[0])
                aabb[1] = np.maximum(aabb[1], batch_aabb[1])

        _1M = min(point_count, 1_000_000)
        steps = math.ceil(point_count / _1M)
        if steps != len(seek_values):
            raise ValueError(
                "the size of seek_values should be equal to steps,"
                f"currently steps = {steps} and len(seek_values) = {len(seek_values)}"
            )
        portions: list[PortionItemType] = [
            (i * _1M, min(point_count, (i + 1) * _1M), seek_values[i])
            for i in range(steps)
        ]

        pointcloud_file_portions = [(path, p) for p in portions]

        has_color = feature_nb >= 6

        # extra fields
        extra_fields = []
        if feature_nb in (4, 7, 8):
            extra_fields.append(
                ExtraFieldsDescription(name="intensity", dtype=np.dtype(np.float32))
            )
        if feature_nb > 7:
            extra_fields.append(
                ExtraFieldsDescription(name="classification", dtype=np.dtype(np.uint32))
            )

    if aabb is None:
        raise ValueError(f"There is no point in the file {path}")

    return {
        "portions": pointcloud_file_portions,
        "aabb": aabb,
        "crs_in": None,
        "point_count": point_count,
        "avg_min": aabb[0],
        "has_color": has_color,
        "extra_fields": extra_fields,
    }


def run(
    filename: Path,
    offset_scale: OffsetScaleType,
    portion: PortionItemType,
    transformer: Transformer | None,
    color_scale: float | None,
    with_rgb: bool,
    extra_fields: list[ExtraFieldsDescription],
) -> Iterator[
    tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.uint8] | None,
        dict[str, npt.NDArray[Any]],
    ],
]:
    with open(filename) as f:

        has_header, feature_nb, delimiter = get_csv_infos(f)

        if has_header:
            f.readline()  # skip first line in case there is a header we promised to ignore

        point_count = portion[1] - portion[0]

        step = min(point_count, max((point_count) // 10, 100000))

        f.seek(portion[2])

        for _ in range(0, point_count, step):
            points = np.zeros((step, feature_nb), dtype=np.float32)

            for j in range(step):
                line = f.readline()
                if not line:
                    points = np.resize(points, (j, feature_nb))
                    break
                points[j] = [float(s) for s in line.split(delimiter)[:feature_nb]]

            x, y, z = (points[:, c] for c in [0, 1, 2])

            if transformer:
                x, y, z = transformer.transform(x, y, z)

            x = (x + offset_scale[0][0]) * offset_scale[1][0]
            y = (y + offset_scale[0][1]) * offset_scale[1][1]
            z = (z + offset_scale[0][2]) * offset_scale[1][2]

            coords = np.vstack((x, y, z)).transpose()

            if offset_scale[2] is not None:
                # Apply transformation matrix (because the tile's transform will contain
                # the inverse of this matrix)
                coords = np.dot(coords, offset_scale[2])

            coords = np.ascontiguousarray(coords.astype(np.float32))

            # Read colors: 3 last columns when excluding classification data
            colors = get_colors(points, feature_nb, color_scale, with_rgb)

            extra_fields_data = {}
            for field in extra_fields:
                if field.name == "intensity":
                    data = points[:, 3]
                elif field.name == "classification":
                    data = points[:, 7]
                else:
                    data = np.zeros(len(points), dtype=field.dtype)
                extra_fields_data[field.name] = np.array(data, dtype=np.uint8)

            yield coords, colors, extra_fields_data
