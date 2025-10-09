import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import numpy.typing as npt
from pyproj import Transformer

from py3dtiles.typing import (
    ExtraFieldsDescription,
    MetadataReaderType,
    OffsetScaleType,
    PortionItemType,
)


def get_metadata(
    filename: Path, color_scale: float | None = None
) -> MetadataReaderType:
    pointcloud_file_portions = []

    with laspy.open(filename) as f:
        point_count = f.header.point_count

        _1M = min(point_count, 1_000_000)
        steps = math.ceil(point_count / _1M)
        portions: list[PortionItemType] = [
            (i * _1M, min(point_count, (i + 1) * _1M)) for i in range(steps)
        ]
        for p in portions:
            pointcloud_file_portions += [(filename, p)]

        crs_in = f.header.parse_crs()

        has_color = "red" in f.header.point_format.dimension_names

        # extra fields
        extra_fields = []
        for fname, the_type in f.header.point_format.dtype().fields.items():
            if fname not in ("X", "Y", "Z", "red", "green", "blue"):
                extra_fields.append(
                    ExtraFieldsDescription(name=fname, dtype=the_type[0])
                )

        # check the file for common errors
        if color_scale is None and has_color:
            points = next(f.chunk_iterator(10_000))
            # we test that the max is > 0 because if all RGB values are 0,
            # we cannot conclude anything about whether or not we have 8 or 16 bit colors
            # the only false positive we would have is if we hit a part of a file that is mostly black
            if (
                0 < np.max(points["red"]) < 256
                and 0 < np.max(points["blue"] < 256)
                and 0 < np.max(points["green"] < 256)
            ):
                print(
                    f"""\
WARNING: the color information in the file {filename} seems to be between 0 and 256  instead of 0 and 65535. You might \
need to set color_scale to 256 if the resulting 3dtiles appears black. See \
https://py3dtiles.org/main/faq.html#when-converting-a-las-laz-files-the-resulting-tiles-are-completely-black \
for more information."""
                )

    return {
        "portions": pointcloud_file_portions,
        "aabb": np.array([f.header.mins, f.header.maxs]),
        "crs_in": crs_in,
        "point_count": point_count,
        "avg_min": np.array(f.header.mins),
        "has_color": has_color,
        "extra_fields": extra_fields,
    }


def run(
    filename: str,
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
    """
    Reads points from a las file
    """
    with laspy.open(filename) as f:

        point_count = portion[1] - portion[0]

        step = min(point_count, max(point_count // 10, 100_000))

        indices = list(range(math.ceil(point_count / step)))

        for index in indices:
            start_offset = portion[0] + index * step
            num = min(step, portion[1] - start_offset)

            # read scaled values and apply offset
            f.seek(start_offset)
            points = next(f.chunk_iterator(num))

            x, y, z = points.x, points.y, points.z
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

            # Read colors
            colors = None
            if with_rgb and "red" in f.header.point_format.dimension_names:
                red = points["red"]
                green = points["green"]
                blue = points["blue"]

                colors = np.vstack((red, green, blue)).transpose()

                if color_scale is not None:
                    colors = np.clip(colors * color_scale, 0, 65535)

                # NOTE las spec says rgb is 16bits by components
                # pnts are 8 bits (by default) by component, hence we divide by 256
                colors = (colors / 256).astype(np.uint8)
            elif with_rgb:
                colors = np.zeros(coords.shape, dtype=np.uint8)

            extra_fields_data = {}
            for extra_field in extra_fields:
                if extra_field.name in f.header.point_format.dimension_names:
                    extra_fields_data[extra_field.name] = points[
                        extra_field.name
                    ].astype(extra_field.dtype)
                else:
                    extra_fields_data[extra_field.name] = np.zeros(
                        len(points), dtype=extra_field.dtype
                    )

            yield coords, colors, extra_fields_data
