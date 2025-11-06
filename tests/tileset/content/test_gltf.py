import numpy as np
import pygltflib
from numpy.testing import assert_array_equal
from pytest import raises

from py3dtiles.points import Points
from py3dtiles.tileset.content.gltf import PointsGltf


def test_points_gltf_from_points() -> None:
    gltf = PointsGltf.from_points(Points(positions=np.zeros((0, 3), dtype=np.float32)))
    assert gltf.get_vertex_count() == 0

    points = Points(
        positions=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        colors=np.array([[256, 0, 0], [0, 0, 128], [0, 128, 0]], dtype=np.uint32),
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
    assert_array_equal(generated_points.colors, points.colors)  # type: ignore [arg-type] # we *know* it's not None
    assert_array_equal(generated_points.extra_fields.keys(), points.extra_fields.keys())  # type: ignore [arg-type] # we *know* it's not None
    assert_array_equal(
        generated_points.extra_fields["intensity"], points.extra_fields["intensity"]
    )


def test_points_gltf_from_gltf() -> None:
    with raises(ValueError, match="Input gltf does not have binary blob"):
        PointsGltf.from_gltf(pygltflib.GLTF2())
    gltf2 = pygltflib.GLTF2()
    gltf2.set_binary_blob(b"foo")
    gltf2.meshes = [pygltflib.Mesh(), pygltflib.Mesh()]
    with raises(ValueError, match="This method supports gltf with exactly one mesh"):
        PointsGltf.from_gltf(gltf2)

    mesh = pygltflib.Mesh()
    mesh.primitives = [pygltflib.Primitive(), pygltflib.Primitive()]
    gltf2.meshes = [mesh]
    with raises(
        ValueError, match="This method supports gltf with exactly one primitive"
    ):
        PointsGltf.from_gltf(gltf2)

    mesh.primitives.pop()
    with raises(ValueError, match="This gltf is not a point gltf, mode is 4"):
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
