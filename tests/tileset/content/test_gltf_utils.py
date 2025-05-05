import re

import numpy as np
import numpy.typing as npt
import pygltflib
from numpy.testing import assert_array_equal
from pytest import raises

from py3dtiles.tileset.content.gltf_utils import (
    GltfMesh,
    GltfPrimitive,
    get_component_type_from_dtype,
    gltf_from_meshes,
    populate_gltf_from_mesh,
)


def test_component_type_from_dtype() -> None:
    assert get_component_type_from_dtype(np.dtype(np.int8)) == pygltflib.BYTE
    assert get_component_type_from_dtype(np.dtype(np.uint8)) == pygltflib.UNSIGNED_BYTE
    assert get_component_type_from_dtype(np.dtype(np.int16)) == pygltflib.SHORT
    assert (
        get_component_type_from_dtype(np.dtype(np.uint16)) == pygltflib.UNSIGNED_SHORT
    )
    assert get_component_type_from_dtype(np.dtype(np.uint32)) == pygltflib.UNSIGNED_INT
    assert get_component_type_from_dtype(np.dtype(np.float32)) == pygltflib.FLOAT
    with raises(
        ValueError,
        match="Cannot find a component type suitable for float64",
    ):
        get_component_type_from_dtype(np.dtype(np.float64))


def test_gtlf_primitive_creation() -> None:
    p1 = GltfPrimitive()
    # triangles is init if not existing
    assert p1.triangles is None
    assert p1.material is None
    assert p1.texture_uri is None

    # assert material and texture_uri are kept
    m = pygltflib.Material()
    p2 = GltfPrimitive(material=m, texture_uri="texture/uri.png")
    assert p2.material == m
    assert p2.texture_uri == "texture/uri.png"

    # assert type of triangles are kept
    p3 = GltfPrimitive(triangles=np.array([1, 2, 3], dtype=np.uint32))
    assert p3.triangles is not None
    assert p3.triangles.component_type == pygltflib.UNSIGNED_INT
    p4 = GltfPrimitive(triangles=np.array([1, 2, 3], dtype=np.uint16))
    assert p4.triangles is not None
    assert p4.triangles.component_type == pygltflib.UNSIGNED_SHORT


def test_gltf_mesh_creation() -> None:
    with raises(
        ValueError,
        match=re.escape(
            "points arguments should be an array of coordinate triplets (of shape (N, 3))"
        ),
    ):
        GltfMesh(np.array([]))
    with raises(
        ValueError,
        match=re.escape(
            "points arguments should be an array of coordinate triplets (of shape (N, 3))"
        ),
    ):
        GltfMesh(np.array([1, 2, 3]))

    GltfMesh(np.array([[1, 2, 3], [4, 5, 6]]))


def test_populate_gltf_from_mesh_simple() -> None:
    gltf = pygltflib.GLTF2()
    m1 = GltfMesh(np.array([[7, 2, 3], [4, 5, 6], [1, 8, 9]], dtype=np.float32))
    assert len(m1.primitives) == 1

    populate_gltf_from_mesh(gltf, m1)

    assert len(gltf.meshes) == 1
    pygltflib_m = gltf.meshes[0]
    assert len(pygltflib_m.primitives) == 1
    p = pygltflib_m.primitives[0]
    assert p.attributes.POSITION is not None
    assert p.indices is None
    assert len(gltf.accessors) == 1
    assert len(gltf.bufferViews) == 1
    assert len(gltf.binary_blob()) == 9 * 4
    assert len(gltf.materials) == 0

    # what if I have a material
    gltf = pygltflib.GLTF2()
    m1.primitives[0].material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[0.5, 0.3, 0.1, 0.9]
        )
    )
    populate_gltf_from_mesh(gltf, m1)
    pygltflib_m = gltf.meshes[0]
    assert pygltflib_m is not None
    assert len(pygltflib_m.primitives) == 1
    p = pygltflib_m.primitives[0]
    assert p.attributes.POSITION is not None
    assert p.indices is None
    assert len(gltf.accessors) == 1
    accessor = gltf.accessors[0]
    assert accessor.min == [1, 2, 3]
    assert accessor.max == [7, 8, 9]
    assert len(gltf.bufferViews) == 1
    assert len(gltf.binary_blob()) == 9 * 4
    assert len(gltf.materials) == 1
    mat = gltf.materials[0]
    mat.pbrMetallicRoughness.baseColorFactor = [0.5, 0.3, 0.1, 0.9]


def test_populate_gltf_from_mesh_one_primitive_with_indices() -> None:
    g = pygltflib.GLTF2()
    p1 = GltfPrimitive(triangles=np.array([[0, 1, 2]], dtype=np.uint8))
    m1 = GltfMesh(
        np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            dtype=np.float32,
        ),
        primitives=[p1],
    )
    # material counter will never be 0 in normal run due to default material
    populate_gltf_from_mesh(g, m1)

    assert len(g.materials) == 0
    assert len(g.meshes) == 1
    pygltflib_m = g.meshes[0]
    assert len(pygltflib_m.primitives) == 1

    # primitive 0
    assert pygltflib_m.primitives[0].material == 0
    # assert buffer, bufferViews, accessors are set correctly for indices
    indices_idx = pygltflib_m.primitives[0].indices
    assert indices_idx == 1
    bv1_idx = g.accessors[indices_idx].bufferView
    assert bv1_idx == 1

    assert_array_equal(
        np.frombuffer(
            g.binary_blob(),
            dtype=np.uint8,
            count=g.accessors[indices_idx].count,
            offset=g.bufferViews[bv1_idx].byteOffset
            + g.accessors[indices_idx].byteOffset,
        ),
        [0, 1, 2],
    )


def test_populate_gltf_from_mesh_several_primitive() -> None:
    g = pygltflib.GLTF2()
    p1 = GltfPrimitive(triangles=np.array([[0, 1, 2]], dtype=np.uint8))
    p2 = GltfPrimitive(triangles=np.array([[2, 1, 3]], dtype=np.uint8))
    m1 = GltfMesh(
        np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [11, 12, 13]],
            dtype=np.float32,
        ),
        primitives=[p1, p2],
    )
    # material counter will never be 0 in normal run due to default material
    populate_gltf_from_mesh(g, m1)

    assert len(g.materials) == 0
    assert len(g.meshes) == 1
    pygltflib_m = g.meshes[0]
    assert len(pygltflib_m.primitives) == 2

    # primitive 0
    assert pygltflib_m.primitives[0].material == 0
    # assert buffer, bufferViews, accessors are set correctly for indices
    indices_idx = pygltflib_m.primitives[0].indices
    assert indices_idx == 1
    bv1_idx = g.accessors[indices_idx].bufferView
    assert bv1_idx == 1
    assert_array_equal(
        np.frombuffer(
            g.binary_blob(),
            dtype=np.uint8,
            count=g.accessors[indices_idx].count,
            offset=g.bufferViews[bv1_idx].byteOffset
            + g.accessors[indices_idx].byteOffset,
        ),
        [0, 1, 2],
    )

    # primitive 1
    assert pygltflib_m.primitives[1].material == 0
    # assert buffer, bufferViews, accessors are set correctly for indices
    indices_idx2 = pygltflib_m.primitives[1].indices
    assert indices_idx2 == 2
    bv2_idx = g.accessors[indices_idx2].bufferView
    assert bv2_idx == 2
    assert_array_equal(
        np.frombuffer(
            g.binary_blob(),
            dtype=np.uint8,
            count=g.accessors[indices_idx2].count,
            offset=g.bufferViews[bv2_idx].byteOffset
            + g.accessors[indices_idx2].byteOffset,
        ),
        [2, 1, 3],
    )

    # test with materials
    g = pygltflib.GLTF2()
    p1.material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1, 0, 0, 0]
        )
    )
    p2.material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[0, 1, 0, 0]
        )
    )
    p3 = GltfPrimitive(
        triangles=np.array([[0, 1, 3]], dtype=np.uint8),
        texture_uri="the_texture/uri.png",
        material=pygltflib.Material(),
    )
    m1.primitives.append(p3)
    populate_gltf_from_mesh(g, m1)
    assert len(g.meshes) == 1
    pygltflib_m = g.meshes[0]
    assert len(pygltflib_m.primitives) == 3
    assert pygltflib_m.primitives[0].material == 0
    assert pygltflib_m.primitives[1].material == 1
    assert pygltflib_m.primitives[2].material == 2

    assert len(g.materials) == 3
    mat1 = g.materials[0]
    assert mat1.pbrMetallicRoughness.baseColorFactor == [1, 0, 0, 0]

    mat2 = g.materials[1]
    assert mat2.pbrMetallicRoughness.baseColorFactor == [0, 1, 0, 0]

    # mat3 with textures
    mat3 = g.materials[2]
    assert mat3.pbrMetallicRoughness.baseColorFactor == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]  # default value for baseColorFactor
    assert mat3.pbrMetallicRoughness.baseColorTexture.index == 0
    assert len(g.textures) == 1
    assert g.textures[0].sampler == 0
    assert len(g.images) == 1
    assert g.images[0].uri == "the_texture/uri.png"


def test_gltf_from_meshes(z_up_matrix: npt.NDArray[np.float32]) -> None:
    mesh1 = GltfMesh(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        normals=np.array([2, 2.5, 3.5], dtype=np.float32),
    )
    mesh2 = GltfMesh(
        np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]], dtype=np.float32),
    )

    m1 = pygltflib.Material()
    m2 = pygltflib.Material(emissiveFactor=[0, 1, 2])
    p1 = GltfPrimitive(
        triangles=np.array([0, 1, 2], dtype=np.uint8),
        material=m1,
        texture_uri="uri/texture.png",
    )
    p2 = GltfPrimitive(triangles=np.array([0, 2, 3], dtype=np.uint8), material=m2)
    mesh3 = GltfMesh(
        np.array(
            [[21, 22, 23], [24, 16, 28], [-1, 2, 5], [12, 999, 2]], dtype=np.float32
        ),
        primitives=[p1, p2],
    )

    # assertions
    gltf = gltf_from_meshes([mesh1, mesh2, mesh3])

    assert len(gltf.nodes) == 3
    assert len(gltf.meshes) == 3
    assert gltf.nodes[0].mesh == 0
    assert gltf.nodes[1].mesh == 1
    assert gltf.nodes[2].mesh == 2

    assert (
        gltf.nodes[0].matrix is None
    )  # by default, we use the implicit identity matrix

    # one for the default material, 2 for the other 2 materials
    assert len(gltf.materials) == 3
    assert len(gltf.images) == 1
    assert gltf.images[0].uri == "uri/texture.png"
    assert len(gltf.samplers) == 1

    # Assert that transformation are correctly propagated
    gltf = gltf_from_meshes([mesh1], transform=z_up_matrix)
    assert len(gltf.nodes) == 1
    assert gltf.nodes[0].matrix == z_up_matrix.flatten("F").tolist()
