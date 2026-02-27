from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy import typing as npt
from pygltflib import GLTF2, POINTS
from typing_extensions import Self

from py3dtiles.exceptions import InvalidGltfError
from py3dtiles.points import Points
from py3dtiles.tileset.content import gltf_utils
from py3dtiles.tileset.content.tile_content import TileContent


@dataclass
class Gltf(TileContent):
    """
    This class wraps a pygltflib.GLTF2 instance, to make it usable to py3dtiles where a TileContent can be used.
    """

    def __init__(self, gltf: GLTF2 | None = None) -> None:
        super().__init__()
        self._gltf = GLTF2() if gltf is None else gltf

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """
        Builds a Gltf from bytes, such as a glb files
        """
        _gltf = GLTF2().load_from_bytes(data)
        return cls(_gltf)

    @classmethod
    def from_meshes(
        cls,
        meshes: list[gltf_utils.GltfMesh],
        transform: npt.NDArray[np.float32] | None = None,
    ) -> Self:
        return cls(gltf_utils.gltf_from_meshes(meshes, transform))

    def to_array(self) -> npt.NDArray[np.uint8]:
        return np.frombuffer(b"".join(self._gltf.save_to_bytes()), dtype=np.uint8)

    def save_as(self, path: Path) -> None:
        """
        Save the gltf to a file
        """
        self._gltf.save(path)

    def get_vertex_count(self) -> int:
        """
        Get the total vertex count. in the file, *without* taking into accounts
        indices. This is therefore different from the total number of vertices
        drawn on-screen.
        """
        return gltf_utils.get_vertex_count(self._gltf)

    def get_vertices(self) -> npt.NDArray[np.float32 | np.uint16] | None:
        """
        Get the vertices in the order they are defined, taking indices into account.
        """
        attribute = gltf_utils.get_attribute(self._gltf, "POSITION")
        return cast(npt.NDArray[np.float32 | np.uint16] | None, attribute)

    def get_colors(self) -> npt.NDArray[np.uint8 | np.uint16 | np.float32] | None:
        """
        Get the colors in the order they are defined, taking indices into account.
        """
        return gltf_utils.get_attribute(self._gltf, "COLOR_0")

    def get_extra_field(self, fieldname: str) -> npt.NDArray[Any] | None:
        """
        Get an extra field in the order they are defined, taking indices into account.
        """
        accessor_name = f"_{fieldname.upper()}"
        return gltf_utils.get_attribute(self._gltf, accessor_name)

    def get_extra_field_names(self) -> set[str]:
        """
        Get the extra field names found in this gltf file. For py3dtiles, an
        extra field is an attribute found on every primitive in this gltf.

        The gltf convention mandates that custom attributes are uppercase and
        starts with a `_`. This method removes this leading `_` and put each
        name to lowercase, to stay consistent with `get_extra_field` and the
        rest of the py3dtiles codebase.
        """
        # TODO replace that with the new metadata property
        return {
            att_name[1:].lower()
            for att_name in gltf_utils.get_non_standard_attribute_names(self._gltf)
        }

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}:\n"
            + f"- num vertices: {str(self.get_vertex_count())}\n"
            + f"- extra fields: {', '.join(self.get_extra_field_names())}\n"
            + f"{self._gltf.to_json()}"
        )


class PointsGltf(Gltf):
    """
    Represents a simple GLTF made uniquely from points and with the following characteristicts:

    - one scene
    - one node
    - one mesh
    - one primitive
    - binary blob

    This is used by py3dtiles to generate pure points gltf. These assumptions makes it vastly easier to implement some operations, like merging 2 PointsGltf together.
    """

    @classmethod
    def from_points(cls, points: Points, name: str | None = None) -> Self:
        if len(points.positions) == 0:
            return cls()
        return cls(gltf_utils.gltf_from_points(points, name))

    @staticmethod
    def can_build_from_gltf(gltf: GLTF2) -> bool:
        # XXX it's certainly possible to work with a gltf that have several meshes and primitives, provided the mode is POINT everywhere
        return (
            gltf.binary_blob() is not None
            and len(gltf.meshes) == 1
            and len(gltf.meshes[0].primitives) == 1
            and gltf.meshes[0].primitives[0].mode == POINTS
        )

    @classmethod
    def from_gltf(cls, gltf: GLTF2) -> Self:
        """
        Builds a PointsGltf from a pygltflib.GLTF2. This checks that the
        assumptions of PointsGltf are respected before constructing the object.
        """
        binary_blob = gltf.binary_blob()
        if binary_blob is None:
            raise InvalidGltfError("Input gltf does not have binary blob.")
        if len(gltf.meshes) != 1:
            raise InvalidGltfError("This method supports gltf with exactly one mesh.")
        mesh = gltf.meshes[0]

        if len(mesh.primitives) != 1:
            raise InvalidGltfError(
                "This method supports gltf with exactly one primitive."
            )

        if mesh.primitives[0].mode != POINTS:
            raise InvalidGltfError(
                f"This gltf is not a point gltf, mode is {mesh.primitives[0].mode}."
            )

        return cls(gltf)

    def to_points(self, transform: npt.NDArray[np.float64] | None) -> Points:
        # positions are always float in gltf
        xyz = cast(
            npt.NDArray[np.float32], gltf_utils.get_attribute(self._gltf, "POSITION")
        )

        rgb = gltf_utils.get_attribute(self._gltf, "COLOR_0")

        extract_fields = {}
        for field in self.get_extra_field_names():
            values = self.get_extra_field(field)
            if values is None:
                raise InvalidGltfError(
                    f"Found no value for field {field} for this gltf."
                )
            extract_fields[field] = values

        points = Points(positions=xyz, colors=rgb, extra_fields=extract_fields)
        if transform is not None:
            points.transform(transform)

        return points

    def merge_with(self, other: Self) -> None:
        """
        Merge 2 gltf built with `gltf_from_points`.

        Warning: this performs a merge according to the structure of the gltf built
        with `gtlf_from_points`. Do not use with other gltf, it will certainly not
        do the correct thing!

        This merges the buffers. It is slower than just concatenating the nodes/meshes, but decreases
        the number of draw call in the viewer, thus making the pointcloud performs better.
        """

        gltf = self._gltf
        other_gltf = other._gltf
        binary_blob = gltf.binary_blob()
        if binary_blob is None:
            raise InvalidGltfError("The current gltf does not have a binary blob??")
        other_binary_blob = other_gltf.binary_blob()
        if other_binary_blob is None:
            # nothing to do
            return
        if len(gltf.meshes) != 1 or len(other_gltf.meshes) != 1:
            raise InvalidGltfError("This method supports gltf with exactly one mesh.")
        mesh = gltf.meshes[0]
        other_mesh = other_gltf.meshes[0]

        if len(mesh.primitives) != 1 or len(other_mesh.primitives) != 1:
            raise InvalidGltfError(
                "This method supports gltf with exactly one primitive."
            )
        primitive = mesh.primitives[0]
        other_primitive = other_mesh.primitives[0]

        for attr in primitive.__dict__:
            if not hasattr(other_primitive, attr):
                raise InvalidGltfError(f"{attr} attribute not found in other.")

        for attr in other_primitive.__dict__:
            if not hasattr(primitive, attr):
                raise InvalidGltfError(f"{attr} attribute not found in self.")

        current_offset = 0
        merged_buffer = b""
        for key, accessor_id in primitive.attributes.__dict__.items():
            if accessor_id is None:
                # GLTF2 includes some default properties, like WEIGHTS_0, None by default
                continue
            accessor = gltf.accessors[accessor_id]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            this_buffer = binary_blob[
                buffer_view.byteOffset : buffer_view.byteOffset + buffer_view.byteLength
            ]

            other_accessor_id = getattr(other_primitive.attributes, key)
            other_accessor = other_gltf.accessors[other_accessor_id]
            other_buffer_view = other_gltf.bufferViews[other_accessor.bufferView]
            other_buffer = other_binary_blob[
                other_buffer_view.byteOffset : other_buffer_view.byteOffset
                + other_buffer_view.byteLength
            ]

            merged_buffer += this_buffer + other_buffer
            accessor.count += other_accessor.count
            buffer_view.byteLength = len(this_buffer) + len(other_buffer)
            buffer_view.byteOffset = current_offset
            current_offset += buffer_view.byteLength

        gltf.set_binary_blob(merged_buffer)
