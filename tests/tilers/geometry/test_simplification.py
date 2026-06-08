import numpy as np
import numpy.typing as npt

from py3dtiles.tilers.geometry.simplification import SimplificationMesh, simplify_mesh


def _make_cube_mesh(
    subdivisions: int = 4, with_uvs: bool = False
) -> SimplificationMesh:
    verts: list[npt.NDArray[np.float32]] = []
    tris: list[int] = []
    uvs: list[list[float]] = []

    def _face(
        a: npt.NDArray[np.float32],
        b: npt.NDArray[np.float32],
        c: npt.NDArray[np.float32],
        d: npt.NDArray[np.float32],
    ) -> None:
        base = len(verts)
        verts.extend([a, b, c, d])
        tris.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        if with_uvs:
            uvs.extend([[0, 0], [0, 1], [1, 0], [1, 1]])

    s = 1.0
    for _ in range(subdivisions):
        faces = [
            ([-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s]),
            ([-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]),
            ([-s, -s, -s], [-s, s, -s], [-s, s, s], [-s, -s, s]),
            ([s, -s, -s], [s, s, -s], [s, s, s], [s, -s, s]),
            ([-s, -s, -s], [s, -s, -s], [s, -s, s], [-s, -s, s]),
            ([-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s]),
        ]
        for a, b, c, d in faces:
            _face(np.array(a), np.array(b), np.array(c), np.array(d))

    return SimplificationMesh(
        vertices=np.array(verts, dtype=np.float64),
        indices=[np.array(tris, dtype=np.int32)],
        uvs=(
            [np.array(uvs, dtype=np.float32), None, None, None]
            if with_uvs
            else [None, None, None, None]
        ),
    )


def test_simplification_simple() -> None:
    # Build a simple subdivided cube (~192 triangles) and halve it.

    cube = _make_cube_mesh(8)
    original_t = cube.triangle_count

    result = simplify_mesh(cube, original_t // 2)
    assert (
        result.triangle_count <= original_t
    ), "Decimation did not reduce triangle count"


def test_simplification_with_uvs() -> None:

    cube = _make_cube_mesh(8, with_uvs=True)
    original_t = cube.triangle_count

    result = simplify_mesh(cube, original_t // 2)
    assert result.triangle_count == 48, "Decimation did not reduce triangle count"
    assert len(result.vertices) == 128, "Decimation did not reduce the vertices count"
    assert result.uvs[0] is not None, "Decimation did not decimate with uvs"
    assert len(result.uvs[0]) == 128, "Decimation did not reduce the uv count"
