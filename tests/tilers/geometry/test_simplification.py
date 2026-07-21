import numpy as np

from py3dtiles.tilers.geometry.simplification import (
    SimplificationMesh,
    SimplificationOptions,
    simplify_mesh,
)


def _make_cube_mesh(
    subdivisions: int = 4, with_uvs: bool = False, with_colors: bool = True
) -> SimplificationMesh:
    vertices: list[float] = []
    triangles: list[int] = []
    texcoords: list[list[float]] = []
    rgba: list[float] = []

    step = 1.0 / subdivisions
    current_index = 0
    for i in range(subdivisions):
        for j in range(subdivisions):
            vertices.extend(
                [
                    i * step,
                    j * step,
                    0,
                    (i + 1) * step,
                    j * step,
                    0,
                    (i + 1) * step,
                    (j + 1) * step,
                    0,
                    (i) * step,
                    (j + 1) * step,
                    0,
                ]
            )
            if with_colors:
                color = (i * j) % 256
                rgba.extend(
                    [
                        color,
                        0,
                        0,
                        1,
                        0,
                        color,
                        0,
                        1,
                        0,
                        0,
                        color,
                        1,
                        color,
                        0,
                        color,
                        1,
                    ]
                )
            triangles.extend(
                [
                    current_index + 0,
                    current_index + 1,
                    current_index + 2,
                    current_index + 0,
                    current_index + 2,
                    current_index + 3,
                ]
            )
            current_index += 4
            if with_uvs:
                texcoords.extend(
                    [
                        [i * step, j * step],
                        [(i + 1) * step, j * step],
                        [(i + 1) * step, (j + 1) * step],
                        [(i) * step, (j + 1) * step],
                    ]
                )

    verts = np.array(vertices, dtype=np.float64).reshape((-1, 3))
    colors = np.array(rgba, dtype=np.float32).reshape((-1, 4))
    tris = np.array(triangles, dtype=np.uint32)
    uvs = np.array(texcoords, dtype=np.float32)

    return SimplificationMesh(
        vertices=verts,
        indices=[tris],
        colors=colors,
        uvs=([uvs, None, None, None] if with_uvs else [None, None, None, None]),
    )


def test_simplification_simple() -> None:
    # Build a simple subdivided cube (~192 triangles) and halve it.

    cube = _make_cube_mesh(8)
    original_t = cube.triangle_count

    result = simplify_mesh(cube, original_t // 2)
    assert result.triangle_count == 63, "Decimation did not reduce triangle count"


def test_simplification_with_uvs() -> None:

    cube = _make_cube_mesh(8, with_uvs=True)
    cube.triangle_count

    result = simplify_mesh(
        cube, 2, SimplificationOptions(aggressiveness=15, preserve_border_edges=True)
    )
    assert result.triangle_count == 2, "Decimation did not reduce triangle count"
    assert len(result.vertices) == 5, "Decimation did not reduce the vertices count"
    assert result.uvs[0] is not None, "Decimation did not decimate with uvs"
    assert len(result.uvs[0]) == 5, "Decimation did not reduce the uv count"


def test_simplification_with_colors() -> None:
    cube = _make_cube_mesh(8, with_uvs=True, with_colors=True)
    cube.triangle_count

    result = simplify_mesh(
        cube,
        2,
        SimplificationOptions(
            aggressiveness=15,
            preserve_border_edges=True,
            preserve_surface_curvature=True,
        ),
    )
    assert result.triangle_count == 2, "Decimation did not reduce triangle count"
    assert len(result.vertices) == 5, "Decimation did not reduce the vertices count"
    assert result.uvs[0] is not None, "Decimation did not decimate with uvs"
    assert len(result.uvs[0]) == 5, "Decimation did not reduce the uv count"
    assert result.colors is not None
    assert len(result.colors) == 5, "Decimation did not reduce the color count"

    # test with invalid colors
    cube.colors = np.array([1, 2, 3])
    result = simplify_mesh(
        cube, 2, SimplificationOptions(aggressiveness=15, preserve_border_edges=True)
    )
    assert result.colors is None, "Invalid colors wasn't ignored"
