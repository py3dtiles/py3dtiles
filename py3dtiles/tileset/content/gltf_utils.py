from __future__ import annotations

from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import pygltflib


class GltfAttribute(NamedTuple):
    """
    A high level representation of a gltf attribute

    `accessor_type` can only take `values autorized by the spec <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_accessor_type>`_.

    `component_type` should take `these values <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_accessor_componenttype>`_.
    """

    name: str
    accessor_type: str  # Literal["SCALAR", "VEC2", "VEC3"] # pygltflib.SCALAR | pygltflib.VEC2 | pygltflib.VEC3
    component_type: pygltflib.UNSIGNED_BYTE | pygltflib.UNSIGNED_INT | pygltflib.FLOAT
    array: npt.NDArray[np.uint8 | np.uint16 | np.uint32 | np.float32]


def get_component_type_from_dtype(dt: np.dtype[Any]) -> int:
    val = None
    if dt == np.int8:
        val = pygltflib.BYTE
    elif dt == np.uint8:
        val = pygltflib.UNSIGNED_BYTE
    elif dt == np.int16:
        val = pygltflib.SHORT
    elif dt == np.uint16:
        val = pygltflib.UNSIGNED_SHORT
    elif dt == np.uint32:
        val = pygltflib.UNSIGNED_INT
    elif dt == np.float32:
        val = pygltflib.FLOAT
    else:
        raise ValueError(f"Cannot find a component type suitable for {dt}")
    return cast(int, val)


class GltfPrimitive:
    """
    A data structure storing all information to create a glTF mesh's primitive.

    This is intended for higher-level usage than pygltflib.Primitive.

    The transformation will be done automatically while transforming a `GltfMesh`.

    :param triangles: array of triangle indices, must have a (n, 3) shape.
    :param material: a glTF material. If not set, a default material is created.
    :param texture_uri: the URI of the texture image if the primitive is textured.
    """

    def __init__(
        self,
        triangles: npt.NDArray[np.uint8 | np.uint16 | np.uint32] | None = None,
        material: pygltflib.Material | None = None,
        texture_uri: str | None = None,
    ) -> None:
        self.triangles: GltfAttribute | None = (
            GltfAttribute(
                "INDICE",
                pygltflib.SCALAR,
                get_component_type_from_dtype(triangles.dtype),
                triangles,
            )
            if triangles is not None
            else None
        )
        self.material: pygltflib.Material | None = material
        self.texture_uri: str | None = texture_uri


class GltfMesh:
    """
    A data structure representing a mesh.

    This is intended for higher-level usage than pygltflib.Mesh, which are an exact translation of the specification.

    This is intented to be easier to construct by keeping a more hierarchical and logical organization. `GltfMesh` are constructed with all the vertices, normals, uvs and additional attributes, and an optional list of `GltfPrimitive` that contains indices and material information.

    Use `gltf_from_meshes` or `populate_gltf_from_mesh` to convert it to GLTF format.

    :param points: array of vertex positions, must have a (n, 3) shape.
    :param primitives: array of GltfPrimitive
    :param normals: array of vertex normals for the whole mesh, must have a (n, 3) shape.
    :param batchids: array of batch table IDs, must have a (n) shape.
    :param additional_attributes: additional attributes to add to the primitive.
    :param uvs: array of texture coordinates, must have a (n, 2) shape.
    """

    def __init__(
        self,
        points: npt.NDArray[np.float32],
        name: str | None = None,
        normals: npt.NDArray[np.float32] | None = None,
        primitives: list[GltfPrimitive] | None = None,
        batchids: npt.NDArray[np.uint32] | None = None,
        uvs: npt.NDArray[np.float32] | None = None,
        additional_attributes: list[GltfAttribute] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        A data structure storing all information to create a glTF mesh's primitive.

        """
        if points is None or len(points.shape) < 2 or points.shape[1] != 3:
            raise ValueError(
                "points arguments should be an array of coordinate triplets (of shape (N, 3))"
            )
        self.points: GltfAttribute = GltfAttribute(
            "POSITION", pygltflib.VEC3, pygltflib.FLOAT, points
        )
        self.name = name
        self.primitives = primitives or []
        if not self.primitives:
            self.primitives.append(GltfPrimitive())
        self.normals: GltfAttribute | None = (
            GltfAttribute("NORMAL", pygltflib.VEC3, pygltflib.FLOAT, normals)
            if normals is not None
            else None
        )
        self.batchids: GltfAttribute | None = (
            GltfAttribute(
                "_BATCHID", pygltflib.SCALAR, pygltflib.UNSIGNED_INT, batchids
            )
            if batchids is not None
            else None
        )
        self.uvs: GltfAttribute | None = (
            GltfAttribute("TEXCOORD_0", pygltflib.VEC2, pygltflib.FLOAT, uvs)
            if uvs is not None
            else None
        )
        self.additional_attributes: list[GltfAttribute] = (
            additional_attributes if additional_attributes is not None else []
        )
        self.properties = properties


def gltf_from_meshes(
    meshes: list[GltfMesh], transform: npt.NDArray[np.float32] | None = None
) -> pygltflib.GLTF2:
    """
    Builds a GLTF2 instance from a list of meshes.
    """

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
    )

    # insert the default material
    # if we don't add it there, we would have to test if one primitive doesn't have one and add
    # only in this case, but we'd need to remember the material_id of the default material.
    # For a few bytes, it's not worthy to do all this.
    # it also makes debugging easier. material_id = 0 is the default material, and that's it.
    # Note that the specification is ambiguous here, it's not clear if a gltf should always have a
    # default material OR if it's the viewer's job.
    gltf.materials.append(pygltflib.Material())

    for i, mesh in enumerate(meshes):
        node = pygltflib.Node(
            mesh=i,
            matrix=(transform.flatten("F").tolist() if transform is not None else None),
        )
        gltf.scenes[0].nodes.append(i)
        gltf.nodes.append(node)

        populate_gltf_from_mesh(
            gltf,
            mesh,
        )

    gltf.buffers = [pygltflib.Buffer(byteLength=len(gltf.binary_blob()))]
    if len(gltf.textures) > 0:
        gltf.samplers.append(
            pygltflib.Sampler(
                magFilter=pygltflib.LINEAR,
                minFilter=pygltflib.LINEAR_MIPMAP_LINEAR,
                wrapS=pygltflib.REPEAT,
                wrapT=pygltflib.REPEAT,
            )
        )

    return gltf


def _set_texture_to_primitive(
    gltf: pygltflib.GLTF2, material: pygltflib.Material, texture_uri: str
) -> None:
    gltf.images.append(pygltflib.Image(uri=texture_uri))
    gltf.textures.append(pygltflib.Texture(sampler=0, source=len(gltf.images) - 1))

    if material.pbrMetallicRoughness is None:
        material.pbrMetallicRoughness = pygltflib.PbrMetallicRoughness()
    if material.pbrMetallicRoughness.baseColorTexture is None:
        material.pbrMetallicRoughness.baseColorTexture = pygltflib.TextureInfo()

    material.pbrMetallicRoughness.baseColorTexture.index = len(gltf.textures) - 1


def _create_gltf_primitive(
    gltf: pygltflib.GLTF2, primitive: GltfPrimitive
) -> pygltflib.Primitive:
    material_id = None
    if primitive.material:

        gltf.materials.append(primitive.material)
        material_id = len(gltf.materials) - 1
    else:
        # default material
        # it will always be 0 by internal py3dtiles convention
        material_id = 0

    return pygltflib.Primitive(
        attributes=pygltflib.Attributes(),
        material=material_id,
    )


def populate_gltf_from_mesh(
    gltf: pygltflib.GLTF2,
    mesh: GltfMesh,
) -> None:
    """
    Add a GltfMesh to a pygltflib.GLTF2

    This method takes care of all the nitty-gritty work about setting buffer, bufferViews, accessors and primitives.
    """
    # see https://gitlab.com/dodgyville/pygltflib/#create-a-mesh
    gltf_binary_blob = cast(bytes, gltf.binary_blob()) or b""

    attributes_array: list[GltfAttribute | None] = [
        mesh.points,
        mesh.normals,
        mesh.uvs,
        mesh.batchids,
    ]
    attributes_array.extend(mesh.additional_attributes)

    attributes_indices_by_name: dict[str, int] = {}
    for attribute in attributes_array:
        if attribute is None:
            continue
        attributes_indices_by_name[attribute.name] = len(gltf.accessors)

        array_blob = prepare_gltf_component(
            gltf,
            attribute,
            len(gltf_binary_blob),
        )

        gltf_binary_blob += array_blob

    gltf_mesh = pygltflib.Mesh(name=mesh.name, extras=mesh.properties)
    for primitive in mesh.primitives:

        # deal with texture
        if primitive.texture_uri:
            # if we have a texture_uri, we need a material to bear it
            if primitive.material is None:
                primitive.material = pygltflib.Material()
            _set_texture_to_primitive(gltf, primitive.material, primitive.texture_uri)
            primitive.material.pbrMetallicRoughness.baseColorTexture.texCoord = (
                attributes_indices_by_name.get("TEXCOORD_0")
            )

        # deal with material
        gltf_primitive = _create_gltf_primitive(gltf, primitive)
        gltf_mesh.primitives.append(gltf_primitive)

        # all primitive shares the same attributes. What they take is determined by the triangles
        # we need to do that because the spec mandates that all attribute accessors of a primitives have the same count
        # this is efficient because we store the attributes data only once
        for name, index in attributes_indices_by_name.items():
            setattr(gltf_primitive.attributes, name, index)
        # triangles
        if primitive.triangles is not None:
            gltf_primitive.indices = len(gltf.accessors)
            indice_blob = prepare_gltf_component(
                gltf,
                primitive.triangles,
                len(gltf_binary_blob),
                pygltflib.ELEMENT_ARRAY_BUFFER,
            )

            gltf_binary_blob += indice_blob

    gltf.meshes.append(gltf_mesh)
    gltf.set_binary_blob(gltf_binary_blob)


def prepare_gltf_component(
    gltf: pygltflib.GLTF2,
    attribute: GltfAttribute,
    byte_offset: int,
    buffer_view_target: int = pygltflib.ARRAY_BUFFER,
) -> bytes:
    array = attribute.array
    array_blob = array.flatten().tobytes()
    # note: triangles are sometimes expressed as array of face vertex indices, but from the gltf point of view, it is a flat scalar array
    count = (
        array.size if attribute.accessor_type == pygltflib.SCALAR else array.shape[0]
    )
    buffer_view = pygltflib.BufferView(
        buffer=0,  # Everything is stored in the same buffer for sake of simplicity
        byteOffset=byte_offset,
        byteLength=len(array_blob),
        target=buffer_view_target,
    )
    gltf.bufferViews.append(buffer_view)
    accessor = pygltflib.Accessor(
        bufferView=len(gltf.bufferViews) - 1,
        componentType=attribute.component_type,
        count=count,
        type=attribute.accessor_type,
    )

    # min / max for positions, mandatory
    if attribute.name == "POSITION":
        accessor.min = np.min(attribute.array, axis=0).tolist()
        accessor.max = np.max(attribute.array, axis=0).tolist()

    gltf.accessors.append(accessor)
    return array_blob
