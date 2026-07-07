import numpy as np
import numpy.typing as npt
import pygltflib
from numpy.testing import assert_array_equal
from pytest import raises

from py3dtiles.exceptions import InvalidGltfError
from py3dtiles.points import Points
from py3dtiles.tileset.content.gltf import PointsGltf


def assert_points_gltf_invariants(gltf: PointsGltf) -> None:
    assert (
        len(gltf._gltf.scenes) == 1
    ), "PointsGltf invariant: expected exactly one scene in the JSON chunk"
    assert (
        len(gltf._gltf.nodes) == 1
    ), "PointsGltf invariant: expected exactly one node in the JSON chunk"
    assert (
        len(gltf._gltf.meshes) == 1
    ), "PointsGltf invariant: expected exactly one mesh in the JSON chunk"
    assert (
        len(gltf._gltf.meshes[0].primitives) == 1
    ), "PointsGltf invariant: expected exactly one primitive in the mesh"
    assert (
        gltf._gltf.binary_blob() is not None
    ), "PointsGltf invariant: expected a single binary chunk"


def test_points_gltf_from_points() -> None:
    gltf = PointsGltf.from_points(Points(positions=np.zeros((0, 3), dtype=np.float32)))
    assert gltf.get_vertex_count() == 0

    points = Points(
        positions=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        colors=np.array([[256, 0, 0], [0, 0, 128], [0, 128, 0]], dtype=np.uint16),
        extra_fields={"intensity": np.array([3, 7, 9], dtype=np.float32)},
    )
    gltf = PointsGltf.from_points(points)

    vertices = gltf.get_vertices()
    assert vertices is not None
    assert_array_equal(vertices, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    colors = gltf.get_colors()
    assert colors is not None
    assert_array_equal(colors, [[256, 0, 0], [0, 0, 128], [0, 128, 0]])
    intensities = gltf.get_extra_field("intensity")
    assert intensities is not None
    assert_array_equal(intensities, [3, 7, 9])

    generated_points = gltf.to_points(None)
    assert_array_equal(generated_points.positions, points.positions)
    assert_array_equal(generated_points.colors, points.colors)
    assert_array_equal(generated_points.extra_fields.keys(), points.extra_fields.keys())
    assert_array_equal(
        generated_points.extra_fields["intensity"], points.extra_fields["intensity"]
    )


def test_points_gltf_from_points_transform(
    z_up_matrix: npt.NDArray[np.float32],
) -> None:
    points = Points(
        positions=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    )
    gltf = PointsGltf.from_points(points)
    positions = np.hstack(
        (
            points.positions,
            np.ones((points.positions.shape[0], 1), dtype=points.positions.dtype),
        )
    )

    assert len(gltf._gltf.nodes) == 1, "Expected a single node in the glTF root"
    node = gltf._gltf.nodes[0]
    transform = np.asarray(node.matrix or np.eye(4), dtype=np.float64)

    transformed = gltf.to_points(transform).positions
    assert_array_equal(
        transformed,
        (positions @ z_up_matrix.T)[:, :3],
        err_msg="Applying node transform should convert positions from Z-up to Y-up",
    )


def test_points_gltf_from_points_nocolor() -> None:
    points = Points(
        positions=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
    )
    gltf = PointsGltf.from_points(points)

    assert_points_gltf_invariants(gltf)
    assert gltf.get_colors() is None, "Expected no COLOR_0 attribute"


def test_points_gltf_from_points_rgb8() -> None:
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    points = Points(
        positions=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        colors=colors,
    )
    gltf = PointsGltf.from_points(points)

    assert_points_gltf_invariants(gltf)
    color0 = gltf._gltf.meshes[0].primitives[0].attributes.COLOR_0
    assert color0 is not None, "Expected COLOR_0 attribute in the primitive"
    assert gltf._gltf.accessors[color0] is not None, "Expected COLOR_0 accessor"

    accessor = gltf._gltf.accessors[color0]
    buffer = gltf.get_colors()
    assert accessor.type == pygltflib.VEC3
    assert accessor.componentType == pygltflib.UNSIGNED_BYTE
    assert accessor.normalized is True
    assert_array_equal(
        buffer,
        colors,
        "Expected colors to be the same as the input colors",
    )


def test_points_gltf_from_points_rgb16() -> None:
    colors = np.array([[65535, 0, 0], [0, 65535, 0], [0, 0, 65535]], dtype=np.uint16)
    points = Points(
        positions=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        colors=colors,
    )
    gltf = PointsGltf.from_points(points)

    assert_points_gltf_invariants(gltf)
    color0 = gltf._gltf.meshes[0].primitives[0].attributes.COLOR_0
    assert color0 is not None, "Expected COLOR_0 attribute in the primitive"
    assert gltf._gltf.accessors[color0] is not None, "Expected COLOR_0 accessor"

    accessor = gltf._gltf.accessors[color0]
    buffer = gltf.get_colors()
    assert accessor.type == pygltflib.VEC3
    assert accessor.componentType == pygltflib.UNSIGNED_SHORT
    assert accessor.normalized is True
    assert_array_equal(
        buffer,
        colors,
        "Expected colors to be the same as the input colors",
    )


def test_points_gltf_from_points_rgb32f() -> None:
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    points = Points(
        positions=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        colors=colors,
    )
    gltf = PointsGltf.from_points(points)

    assert_points_gltf_invariants(gltf)
    color0 = gltf._gltf.meshes[0].primitives[0].attributes.COLOR_0
    assert color0 is not None, "Expected COLOR_0 attribute in the primitive"
    assert gltf._gltf.accessors[color0] is not None, "Expected COLOR_0 accessor"

    accessor = gltf._gltf.accessors[color0]
    buffer = gltf.get_colors()
    assert accessor.type == pygltflib.VEC3
    assert accessor.componentType == pygltflib.FLOAT
    assert accessor.normalized is False or None

    assert_array_equal(
        buffer,
        colors,
        "Expected buffer view to be the same as the input colors",
    )


def test_points_gltf_from_gltf() -> None:
    with raises(InvalidGltfError, match="Input gltf does not have binary blob"):
        PointsGltf.from_gltf(pygltflib.GLTF2())
    gltf2 = pygltflib.GLTF2()
    gltf2.set_binary_blob(b"foo")
    gltf2.meshes = [pygltflib.Mesh(), pygltflib.Mesh()]
    with raises(
        InvalidGltfError, match="This method supports gltf with exactly one mesh"
    ):
        PointsGltf.from_gltf(gltf2)

    mesh = pygltflib.Mesh()
    mesh.primitives = [pygltflib.Primitive(), pygltflib.Primitive()]
    gltf2.meshes = [mesh]
    with raises(
        InvalidGltfError, match="This method supports gltf with exactly one primitive"
    ):
        PointsGltf.from_gltf(gltf2)

    mesh.primitives.pop()
    with raises(InvalidGltfError, match="This gltf is not a point gltf, mode is 4"):
        PointsGltf.from_gltf(gltf2)

    mesh.primitives[0].mode = pygltflib.POINTS
    assert PointsGltf.from_gltf(gltf2) is not None


def test_gltfpoints_from_points() -> None:
    pts = Points(
        positions=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32),
        colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint32),
        extra_fields={"field1": np.array([3, 5, 7], dtype=np.uint32)},
    )
    g = PointsGltf.from_points(pts, "foo")
    assert g.get_vertex_count() == 3

    assert len(g._gltf.meshes) == 1
    m = g._gltf.meshes[0]
    assert len(m.primitives) == 1
    p = m.primitives[0]
    assert p.mode == pygltflib.POINTS
    position_accessor = g._gltf.accessors[p.attributes.POSITION]
    assert position_accessor.count == 3
    assert_array_equal(
        np.frombuffer(
            g._gltf.binary_blob(),
            dtype=np.float32,
            count=position_accessor.count * 3,
            offset=g._gltf.bufferViews[position_accessor.bufferView].byteOffset
            + position_accessor.byteOffset,
        ).reshape((-1, 3)),
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
    )

    color_accessor = g._gltf.accessors[p.attributes.COLOR_0]
    assert color_accessor.count == 3
    assert_array_equal(
        np.frombuffer(
            g._gltf.binary_blob(),
            dtype=np.uint32,
            count=color_accessor.count * 3,
            offset=g._gltf.bufferViews[color_accessor.bufferView].byteOffset
            + color_accessor.byteOffset,
        ).reshape((-1, 3)),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    field_accessor = g._gltf.accessors[p.attributes._FIELD1]
    assert field_accessor.count == 3
    assert_array_equal(
        np.frombuffer(
            g._gltf.binary_blob(),
            # py3dtiles automatically switch that to float32, because uint32 are forbidden for custom attributes
            dtype=np.float32,
            count=field_accessor.count,
            offset=g._gltf.bufferViews[field_accessor.bufferView].byteOffset
            + field_accessor.byteOffset,
        ),
        [3, 5, 7],
    )


def test_merge_points_gltf(points_gltf1: PointsGltf, points_gltf2: PointsGltf) -> None:

    points_gltf1.merge_with(points_gltf2)
    g = points_gltf1._gltf

    assert points_gltf1.get_vertex_count() == 6

    assert len(g.meshes) == 1
    m = g.meshes[0]
    assert len(m.primitives) == 1
    p = m.primitives[0]
    assert p.mode == pygltflib.POINTS
    position_accessor = g.accessors[p.attributes.POSITION]
    assert position_accessor.count == 6
    assert_array_equal(
        np.frombuffer(
            g.binary_blob(),
            dtype=np.float32,
            count=position_accessor.count * 3,
            offset=g.bufferViews[position_accessor.bufferView].byteOffset
            + position_accessor.byteOffset,
        ).reshape((-1, 3)),
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, -1, 1], [2, -2, 2], [3, -3, 3]],
    )

    color_accessor = g.accessors[p.attributes.COLOR_0]
    assert color_accessor.count == 6
    assert_array_equal(
        np.frombuffer(
            g.binary_blob(),
            dtype=np.uint32,
            count=color_accessor.count * 3,
            offset=g.bufferViews[color_accessor.bufferView].byteOffset
            + color_accessor.byteOffset,
        ).reshape((-1, 3)),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [256, 0, 0], [0, 256, 0], [0, 0, 256]],
    )

    field_accessor = g.accessors[p.attributes._FIELD1]
    assert field_accessor.count == 6
    assert_array_equal(
        np.frombuffer(
            g.binary_blob(),
            # py3dtiles automatically switch that to float32, because uint32 are forbidden for custom attributes
            dtype=np.float32,
            count=field_accessor.count,
            offset=g.bufferViews[field_accessor.bufferView].byteOffset
            + field_accessor.byteOffset,
        ),
        [3, 5, 7, 11, 13, 17],
    )
