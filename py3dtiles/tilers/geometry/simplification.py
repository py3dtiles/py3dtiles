"""
Fast Quadric Mesh Simplification - Python/NumPy Port

Original C# implementation by Mattias Edlund (MIT License)
Based on "Mesh Simplification Tutorial" (C) Sven Forstmann 2014 (MIT License)
https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification

This python port has been done by Claude (Sonnet 4.6) from this file:
https://github.com/OpenDroneMap/Obj2Tiles/blob/b6c5f02/MeshDecimatorCore/Algorithms/FastQuadricMeshSimplification.cs

Claude claimed to have preserved all original logic including:
  - Quadric error metrics
  - Border / seam / foldover vertex detection
  - Smart linking of near-coincident border vertices
  - Surface curvature preservation
  - UV / normal / color attribute interpolation
  - Lossless and lossy decimation modes
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy import typing as npt

# ---------------------------------------------------------------------------
# Symmetric 4x4 matrix (upper-triangle only, 10 coefficients)
# ---------------------------------------------------------------------------


class SymmetricMatrix:
    """
    Represents the upper triangle of a symmetric 4×4 matrix used for
    quadric error computation.

    Layout (row-major upper triangle):
        m0  m1  m2  m3
            m4  m5  m6
                m7  m8
                    m9
    """

    __slots__ = ("m",)

    def __init__(self, a: float = 0, b: float = 0, c: float = 0, d: float = 0):
        """
        When called with four args (a, b, c, d) initialises the matrix as the
        outer product n·nᵀ for the plane equation ax+by+cz+d=0.
        """
        self.m = np.array(
            [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    def __iadd__(self, other: SymmetricMatrix) -> SymmetricMatrix:
        self.m += other.m
        return self

    def __add__(self, other: SymmetricMatrix) -> SymmetricMatrix:
        result = SymmetricMatrix()
        result.m = self.m + other.m
        return result

    # ------------------------------------------------------------------
    # Sub-determinants used by CalculateError
    # ------------------------------------------------------------------
    def det1(self) -> float:
        """det of the top-left 3×3 sub-matrix."""
        m = self.m
        # we know the array has enough elements, let's override the float|Any return type
        return cast(
            float,
            m[0] * m[4] * m[7]
            + 2.0 * m[1] * m[2] * m[5]
            - m[2] * m[2] * m[4]
            - m[0] * m[5] * m[5]
            - m[1] * m[1] * m[7],
        )

    def det2(self) -> float:
        m = self.m
        return cast(
            float,
            m[1] * m[5] * m[8]
            + m[2] * m[6] * m[5]
            + m[3] * m[2] * m[7]  # sign flip vs det1 layout
            - m[3] * m[5] * m[5]
            - m[1] * m[6] * m[8]  # actually  Determinant2 in original
            - m[2] * m[2] * m[6],
        )

    def det3(self) -> float:
        m = self.m
        return cast(
            float,
            m[0] * m[5] * m[8]
            + m[1] * m[3] * m[7]
            + m[2] * m[1] * m[8]
            - m[2] * m[5] * m[3]
            - m[0] * m[6] * m[7]
            - m[1] * m[1] * m[8],
        )

    def det4(self) -> float:
        m = self.m
        return cast(
            float,
            m[0] * m[4] * m[8]
            + 2.0 * m[1] * m[2] * m[6]
            - m[2] * m[2] * m[4]
            - m[0] * m[6] * m[6]
            - m[1] * m[1] * m[8],
        )


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class SimplificationOptions:
    preserve_border_edges: bool = False
    preserve_uv_seam_edges: bool = False
    preserve_uv_foldover_edges: bool = False
    preserve_surface_curvature: bool = False
    enable_smart_link: bool = True
    vertex_link_distance: float = float(np.finfo(np.float64).eps) * 100
    max_iteration_count: int = 100
    aggressiveness: float = 7.0
    verbose: bool = False


# ---------------------------------------------------------------------------
# Mesh container
# ---------------------------------------------------------------------------


@dataclass
class SimplificationMesh:
    """Simple triangle-mesh container."""

    vertices: npt.NDArray[np.float64]  # (V, 3) float64
    indices: list[npt.NDArray[np.uint32]]  # one int32 array per sub-mesh, length = 3*T
    normals: npt.NDArray[np.float32] | None = None  # (V, 3) float32
    colors: npt.NDArray[np.float32] | None = None  # (V, 4) float32
    uvs: list[npt.NDArray[np.float32] | None] = field(
        default_factory=lambda: [None] * 4
    )  # (V, 2|3|4)

    # NOTE, the original algorithm supports we will use only channel 0 for now.
    UV_CHANNEL_COUNT = 4

    @property
    def sub_mesh_count(self) -> int:
        return len(self.indices)

    @property
    def triangle_count(self) -> int:
        return sum(len(idx) // 3 for idx in self.indices)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


class FastQuadricMeshSimplification:
    """
    Fast Quadric Mesh Simplification.

    Usage
    -----
    sim = FastQuadricMeshSimplification(options)
    sim.initialize(mesh)
    sim.decimate_mesh(target_triangle_count)   # or decimate_mesh_lossless()
    result = sim.to_mesh()
    """

    _DOUBLE_EPSILON = 1.0e-3

    # ------------------------------------------------------------------
    def __init__(self, options: SimplificationOptions | None = None):
        self.options = options or SimplificationOptions()
        self._reset()

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        # Vertex arrays
        self._vertex_positions: list[npt.NDArray[np.float64]] = []
        # accumulated quadric error matrices
        self._vertex_quadrics: list[SymmetricMatrix] = []  # quadrics
        # The ref arrays are actually a lookup table to know which triangle touch vertex i
        # start index into the ref list
        self._vertex_ref_start: list[int] = []
        # how many refs (= how many triangle) touch this vertex
        self._vertex_ref_count: list[int] = []
        # vertices on border, mesh boundary
        self._v_border: list[bool] = []
        # coincident border verts with different uvs
        self._v_seam: list[bool] = []
        # coincident border verts with same uvs
        self._v_foldover: list[bool] = []

        # Triangle arrays (all parallel lists)
        self._triangle_vertex_indices: list[npt.NDArray[np.int32]] = (
            []
        )  # shape (3,) int  – vertex indices
        self._triangle_attribute_indices: list[npt.NDArray[np.int32]] = (
            []
        )  # shape (3,) int  – attribute indices
        self._triangle_edge_errors: list[npt.NDArray[np.float64]] = (
            []
        )  # shape (4,) float – err0..err3
        self._triangle_deleted: list[bool] = []
        self._triangle_dirty: list[bool] = []
        self._triangle_normals: list[npt.NDArray[np.float64]] = []  # shape (3,)
        self._triangle_sub_mesh: list[int] = []

        # Ref arrays
        self._ref_triangle_id: list[int] = []
        self._ref_corner_index: list[int] = []

        # Vertex attributes
        self._va_normals: list[npt.NDArray[np.float32]] | None = None
        self._va_uvs: list[list[npt.NDArray[np.float32]] | None] = [
            None
        ] * SimplificationMesh.UV_CHANNEL_COUNT
        self._va_uv_dims: list[int] = [0] * SimplificationMesh.UV_CHANNEL_COUNT
        self._va_colors: list[npt.NDArray[np.float32]] | None = None

        self._sub_mesh_count = 0
        self._remaining_vertices = 0

        self._vert_curvatures: list[float] | None = None

    # ===================================================================
    # Public API
    # ===================================================================

    def initialize(self, mesh: SimplificationMesh) -> None:
        """Load mesh data into internal structures."""
        self._reset()
        self._sub_mesh_count = mesh.sub_mesh_count

        # Vertices
        for p in mesh.vertices:
            self._vertex_positions.append(np.array(p, dtype=np.float64))
            self._vertex_quadrics.append(SymmetricMatrix())
            self._vertex_ref_start.append(0)
            self._vertex_ref_count.append(0)
            self._v_border.append(True)
            self._v_seam.append(False)
            self._v_foldover.append(False)

        vertices_count = len(self._vertex_positions)

        # Triangles
        for sub_idx, sub_indices in enumerate(mesh.indices):
            for i in range(0, len(sub_indices), 3):
                v0, v1, v2 = (
                    int(sub_indices[i]),
                    int(sub_indices[i + 1]),
                    int(sub_indices[i + 2]),
                )
                self._triangle_vertex_indices.append(
                    np.array([v0, v1, v2], dtype=np.int32)
                )
                self._triangle_attribute_indices.append(
                    np.array([v0, v1, v2], dtype=np.int32)
                )
                self._triangle_edge_errors.append(np.zeros(4, dtype=np.float64))
                self._triangle_deleted.append(False)
                self._triangle_dirty.append(False)
                # TODO shouldn't we initialize with mesh.normals? What's the diff with _va_normals?
                self._triangle_normals.append(np.zeros(3, dtype=np.float64))
                self._triangle_sub_mesh.append(sub_idx)

        # Attributes
        # TODO extract that elsewhere ?
        # TODO use numpy.copy instead?
        def _init_attr(
            arr: npt.NDArray[np.float32] | None, name: str, expected_len: int
        ) -> list[npt.NDArray[np.float32]] | None:
            if arr is not None:
                if len(arr) == expected_len:
                    return [
                        np.array(arr[i], dtype=np.float32) for i in range(expected_len)
                    ]
                elif len(arr) > 0:
                    print(
                        f"Warning: attribute '{name}' has {len(arr)} elements, expected {expected_len}"
                    )
            return None

        self._va_normals = _init_attr(mesh.normals, "normals", vertices_count)
        self._va_colors = _init_attr(mesh.colors, "colors", vertices_count)

        for ch in range(SimplificationMesh.UV_CHANNEL_COUNT):
            uv = mesh.uvs[ch] if ch < len(mesh.uvs) else None
            if uv is not None and len(uv) == vertices_count:
                dim = np.array(uv[0]).shape[0] if hasattr(uv[0], "__len__") else 2
                self._va_uvs[ch] = [
                    np.array(uv[i], dtype=np.float32) for i in range(vertices_count)
                ]
                self._va_uv_dims[ch] = dim
            else:
                self._va_uvs[ch] = None
                self._va_uv_dims[ch] = 0

    # ------------------------------------------------------------------
    def decimate_mesh(self, target_tris_count: int) -> None:
        """Lossy decimation down to target_tris_count triangles."""
        if target_tris_count < 0:
            raise ValueError("target_tris_count must be >= 0")

        opts = self.options
        deleted_tris_count = 0
        triangle_count = len(self._triangle_vertex_indices)
        start_tris = triangle_count

        for iteration in range(opts.max_iteration_count):
            current = start_tris - deleted_tris_count
            if opts.verbose and (iteration % 5) == 0:
                print(f"  iteration {iteration} - triangles {current}")

            if current <= target_tris_count:
                break

            if (iteration % 5) == 0:
                self._update_mesh(iteration)

            # Clear dirty
            for i in range(len(self._triangle_vertex_indices)):
                self._triangle_dirty[i] = False

            threshold = 1e-9 * math.pow(iteration + 3, opts.aggressiveness)

            deleted0: list[bool] = []
            deleted1: list[bool] = []
            deleted_tris_count = self._remove_vertex_pass(
                start_tris,
                target_tris_count,
                threshold,
                deleted0,
                deleted1,
                deleted_tris_count=deleted_tris_count,
            )

        self._compact_mesh()

    # ------------------------------------------------------------------
    def decimate_mesh_lossless(self) -> None:
        """Lossless decimation – removes only zero-error edges."""
        start_tris = len(self._triangle_vertex_indices)

        for iteration in range(self.opts.max_iteration_count):
            self._update_mesh(iteration)

            for i in range(len(self._triangle_vertex_indices)):
                self._triangle_dirty[i] = False

            if self.options.verbose:
                print(f"  Lossless iteration {iteration}")

            deleted_tris_count = self._remove_vertex_pass(
                start_tris, 0, self._DOUBLE_EPSILON, [], [], deleted_tris_count=0
            )

            if deleted_tris_count <= 0:
                break

        self._compact_mesh()

    # ------------------------------------------------------------------
    def to_mesh(self) -> SimplificationMesh:
        """Convert internal state back to a Mesh."""
        vertices_count = len(self._vertex_positions)
        triangles_count = len(self._triangle_vertex_indices)

        vertices = np.array([self._vertex_positions[i] for i in range(vertices_count)])

        # Sub-mesh index splits
        sub_offsets = [0] * self._sub_mesh_count
        last_sub = -1
        for i in range(triangles_count):
            s = self._triangle_sub_mesh[i]
            if s != last_sub:
                for j in range(last_sub + 1, s):
                    sub_offsets[j] = i
                sub_offsets[s] = i
                last_sub = s
        for i in range(last_sub + 1, self._sub_mesh_count):
            sub_offsets[i] = triangles_count

        indices_out = []
        for s in range(self._sub_mesh_count):
            start = sub_offsets[s]
            end = (
                sub_offsets[s + 1] if s + 1 < self._sub_mesh_count else triangles_count
            )
            idx = []
            for ti in range(start, end):
                v = self._triangle_vertex_indices[ti]
                idx.extend([int(v[0]), int(v[1]), int(v[2])])
            indices_out.append(np.array(idx, dtype=np.uint32))

        new_mesh = SimplificationMesh(vertices=vertices, indices=indices_out)

        if self._va_normals is not None:
            new_mesh.normals = np.array(self._va_normals[:vertices_count])
        if self._va_colors is not None:
            new_mesh.colors = np.array(self._va_colors[:vertices_count])

        uvs_out: list[npt.NDArray[np.float32] | None] = [
            None
        ] * SimplificationMesh.UV_CHANNEL_COUNT
        for ch in range(SimplificationMesh.UV_CHANNEL_COUNT):
            current_uvs_channel = self._va_uvs[ch]
            if current_uvs_channel is not None:
                uvs_out[ch] = np.array(current_uvs_channel[:vertices_count])
        new_mesh.uvs = uvs_out

        return new_mesh

    # ===================================================================
    # Private helpers
    # ===================================================================

    def _vertex_error(
        self, q: SymmetricMatrix, x: float, y: float, z: float
    ) -> np.float64:
        m = q.m
        # for some reason, mypy thinks it's Any. Maybe because of the indexing?
        return cast(
            np.float64,
            (
                m[0] * x * x
                + 2 * m[1] * x * y
                + 2 * m[2] * x * z
                + 2 * m[3] * x
                + m[4] * y * y
                + 2 * m[5] * y * z
                + 2 * m[6] * y
                + m[7] * z * z
                + 2 * m[8] * z
                + m[9]
            ),
        )

    def _calculate_error(
        self, i0: int, i1: int
    ) -> tuple[float, npt.NDArray[np.float64], int]:
        """
        Returns (error, result_point, result_index).
        result_index: 0=p1, 1=p2, 2=midpoint-or-optimal
        """
        q = self._vertex_quadrics[i0] + self._vertex_quadrics[i1]
        border = self._v_border[i0] and self._v_border[i1]
        det = q.det1()
        if det != 0.0 and not border:
            result = np.array(
                [-1.0 / det * q.det2(), 1.0 / det * q.det3(), -1.0 / det * q.det4()]
            )
            error = self._vertex_error(q, result[0], result[1], result[2])
            return error, result, 2
        else:
            p1 = self._vertex_positions[i0]
            p2 = self._vertex_positions[i1]
            p3 = (p1 + p2) * 0.5
            e1 = self._vertex_error(q, p1[0], p1[1], p1[2])
            e2 = self._vertex_error(q, p2[0], p2[1], p2[2])
            e3 = self._vertex_error(q, p3[0], p3[1], p3[2])
            error = min(e1, e2, e3)
            if error == e3:
                return error, p3, 2
            elif error == e2:
                return error, p2, 1
            else:
                return error, p1, 0

    def _calculate_error_with_curvature(
        self, i0: int, i1: int
    ) -> tuple[float, npt.NDArray[np.float64], int]:
        error, result, idx = self._calculate_error(i0, i1)
        if hasattr(self, "_vert_curvatures") and self._vert_curvatures is not None:
            curvature = max(self._vert_curvatures[i0], self._vert_curvatures[i1])
            error += error * curvature
        return error, result, idx

    # ------------------------------------------------------------------
    def _flipped(
        self, point: npt.NDArray[np.float64], i0: int, i1: int, deleted: list[bool]
    ) -> bool:
        tstart = self._vertex_ref_start[i0]
        tcount = self._vertex_ref_count[i0]
        for k in range(tcount):
            tid = self._ref_triangle_id[tstart + k]
            if self._triangle_deleted[tid]:
                continue
            s = self._ref_corner_index[tstart + k]
            id1 = self._triangle_vertex_indices[tid][(s + 1) % 3]
            id2 = self._triangle_vertex_indices[tid][(s + 2) % 3]
            if id1 == i1 or id2 == i1:
                deleted[k] = True
                continue

            d1 = self._vertex_positions[id1] - point
            d1_norm = np.linalg.norm(d1)
            if d1_norm > 0:
                d1 /= d1_norm
            d2 = self._vertex_positions[id2] - point
            d2_norm = np.linalg.norm(d2)
            if d2_norm > 0:
                d2 /= d2_norm

            if abs(np.dot(d1, d2)) > 0.999:
                return True

            n = np.cross(d1, d2)
            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                n /= n_norm
            deleted[k] = False
            if np.dot(n, self._triangle_normals[tid]) < 0.2:
                return True
        return False

    # ------------------------------------------------------------------
    def _update_triangles(
        self,
        i0: int,
        ia0: int,
        v_idx: int,
        deleted: list[bool],
    ) -> tuple[list[tuple[int, int]], int]:
        tstart = self._vertex_ref_start[v_idx]
        tcount = self._vertex_ref_count[v_idx]
        new_refs = []
        deleted_tris_count = 0
        for k in range(tcount):
            rid = tstart + k
            tid = self._ref_triangle_id[rid]
            tv = self._ref_corner_index[rid]
            if self._triangle_deleted[tid]:
                continue
            if deleted[k]:
                self._triangle_deleted[tid] = True
                deleted_tris_count += 1
                continue

            self._triangle_vertex_indices[tid][tv] = i0
            if ia0 != -1:
                self._triangle_attribute_indices[tid][tv] = ia0

            self._triangle_dirty[tid] = True
            e0, _, _ = self._calculate_error_with_curvature(
                int(self._triangle_vertex_indices[tid][0]),
                int(self._triangle_vertex_indices[tid][1]),
            )
            e1, _, _ = self._calculate_error_with_curvature(
                int(self._triangle_vertex_indices[tid][1]),
                int(self._triangle_vertex_indices[tid][2]),
            )
            e2, _, _ = self._calculate_error_with_curvature(
                int(self._triangle_vertex_indices[tid][2]),
                int(self._triangle_vertex_indices[tid][0]),
            )
            self._triangle_edge_errors[tid][0] = e0
            self._triangle_edge_errors[tid][1] = e1
            self._triangle_edge_errors[tid][2] = e2
            self._triangle_edge_errors[tid][3] = min(e0, e1, e2)
            new_refs.append((tid, tv))

        return new_refs, deleted_tris_count

    # ------------------------------------------------------------------
    @staticmethod
    def _barycentric(
        point: npt.NDArray[np.float64],
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        eps = 1e-8
        v0 = b - a
        v1 = c - a
        v2 = point - a
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < eps:
            denom = eps
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    def _interpolate_vertex_attributes(
        self, dst: int, i0: int, i1: int, i2: int, point: npt.NDArray[np.float64]
    ) -> None:
        u, v, w = self._barycentric(
            point,
            self._vertex_positions[i0],
            self._vertex_positions[i1],
            self._vertex_positions[i2],
        )
        fu, fv, fw = float(u), float(v), float(w)

        if self._va_normals is not None:
            n = self._va_normals
            result = n[i0] * fu + n[i1] * fv + n[i2] * fw
            norm = np.linalg.norm(result)
            if norm > 0:
                result /= norm
            n[dst] = result

        for ch in range(SimplificationMesh.UV_CHANNEL_COUNT):
            current_ch = self._va_uvs[ch]
            if current_ch is not None:
                current_ch[dst] = (
                    current_ch[i0] * fu + current_ch[i1] * fv + current_ch[i2] * fw
                )

        if self._va_colors is not None:
            c = self._va_colors
            c[dst] = c[i0] * fu + c[i1] * fv + c[i2] * fw

    def _are_uvs_same(self, channel: int, a: int, b: int) -> bool:
        uv = self._va_uvs[channel]
        if uv is not None:
            return np.array_equal(uv[a], uv[b])
        return False

    # ------------------------------------------------------------------
    def _calculate_vertex_curvatures(self) -> None:
        vertices_count = len(self._vertex_positions)
        curvatures = [0.0] * vertices_count
        for i in range(vertices_count):
            tstart = self._vertex_ref_start[i]
            tcount = self._vertex_ref_count[i]
            if tcount <= 1:
                continue
            max_curv = 0.0
            for j in range(tcount):
                tidA = self._ref_triangle_id[tstart + j]
                if self._triangle_deleted[tidA]:
                    continue
                nA = self._triangle_normals[tidA]
                for k in range(j + 1, tcount):
                    tidB = self._ref_triangle_id[tstart + k]
                    if self._triangle_deleted[tidB]:
                        continue
                    nB = self._triangle_normals[tidB]
                    dot = float(np.dot(nA, nB))
                    dot = max(-1.0, min(1.0, dot))
                    curv = (1.0 - dot) * 0.5
                    if curv > max_curv:
                        max_curv = curv
            curvatures[i] = max_curv
        self._vert_curvatures = curvatures

    # ------------------------------------------------------------------
    def _remove_vertex_pass(
        self,
        start_tris: int,
        target_tris: int,
        threshold: float,
        deleted0: list[bool],
        deleted1: list[bool],
        deleted_tris_count: int,
    ) -> int:
        opts = self.options
        triangle_count = len(self._triangle_vertex_indices)

        flipped_count = 0
        error_too_big_count = 0
        because_border = 0
        because_seam = 0
        because_foldover = 0
        for tid in range(triangle_count):
            if (
                self._triangle_dirty[tid]
                or self._triangle_deleted[tid]
                or self._triangle_edge_errors[tid][3] > threshold
            ):
                continue

            for edge_idx in range(3):
                if self._triangle_edge_errors[tid][edge_idx] > threshold:
                    error_too_big_count += 1
                    continue

                next_edge = (edge_idx + 1) % 3
                i0 = int(self._triangle_vertex_indices[tid][edge_idx])
                i1 = int(self._triangle_vertex_indices[tid][next_edge])

                if self._v_border[i0] != self._v_border[i1]:
                    because_border += 1
                    continue
                if self._v_seam[i0] != self._v_seam[i1]:
                    because_seam += 1
                    continue
                if self._v_foldover[i0] != self._v_foldover[i1]:
                    because_foldover += 1
                    continue
                if opts.preserve_border_edges and self._v_border[i0]:
                    continue
                if opts.preserve_uv_seam_edges and self._v_seam[i0]:
                    continue
                if opts.preserve_uv_foldover_edges and self._v_foldover[i0]:
                    continue

                _, p, _ = self._calculate_error_with_curvature(i0, i1)
                # if iteration > 10:
                #     print("error", threshold, p)

                tc0 = self._vertex_ref_count[i0]
                tc1 = self._vertex_ref_count[i1]
                deleted0.clear()
                deleted0.extend([False] * tc0)
                deleted1.clear()
                deleted1.extend([False] * tc1)

                if self._flipped(p, i0, i1, deleted0):
                    flipped_count += 1
                    continue
                if self._flipped(p, i1, i0, deleted1):
                    flipped_count += 1
                    continue

                ia0 = int(self._triangle_attribute_indices[tid][edge_idx])
                ia1 = int(self._triangle_attribute_indices[tid][next_edge])
                third_edge = 3 - edge_idx - next_edge
                ia2 = int(self._triangle_attribute_indices[tid][third_edge])
                self._interpolate_vertex_attributes(ia0, ia0, ia1, ia2, p)

                # Collapse edge: move i0 to p, absorb i1's quadric
                self._vertex_positions[i0] = p.copy()
                self._vertex_quadrics[i0] += self._vertex_quadrics[i1]

                effective_ia0 = -1 if self._v_seam[i0] else ia0

                len(self._ref_triangle_id)
                new_refs0, deleted_count0 = self._update_triangles(
                    i0, effective_ia0, i0, deleted0
                )
                deleted_tris_count += deleted_count0
                new_refs1, deleted_count1 = self._update_triangles(
                    i0, effective_ia0, i1, deleted1
                )
                deleted_tris_count += deleted_count1
                all_new = new_refs0 + new_refs1
                tcount_new = len(all_new)

                old_tstart = self._vertex_ref_start[i0]
                old_tcount = self._vertex_ref_count[i0]

                if tcount_new <= old_tcount:
                    # overwrite in place
                    for k, (t, tv) in enumerate(all_new):
                        self._ref_triangle_id[old_tstart + k] = t
                        self._ref_corner_index[old_tstart + k] = tv
                else:
                    # append
                    new_start = len(self._ref_triangle_id)
                    for t, tv in all_new:
                        self._ref_triangle_id.append(t)
                        self._ref_corner_index.append(tv)
                    self._vertex_ref_start[i0] = new_start

                self._vertex_ref_count[i0] = tcount_new
                self._remaining_vertices -= 1
                break

            current = start_tris - deleted_tris_count
            # TODO condition always true (second part)
            if current <= target_tris:
                if opts.verbose:
                    print(
                        f"breaking because current is {current}, target_tris is {target_tris} (remaining vertices is) ${self._remaining_vertices}"
                    )
                break
        if opts.verbose:
            print("STAT")
            print("flipped_count", flipped_count)
            print("error_too_big_count", error_too_big_count)
            print("because_border", because_border)
            print("because_seam", because_seam)
            print("because_foldover", because_foldover)

        return deleted_tris_count

    # ------------------------------------------------------------------
    def _update_mesh(self, iteration: int) -> None:
        triangle_count = len(self._triangle_vertex_indices)
        vertex_count = len(self._vertex_positions)

        if iteration > 0:
            # Compact deleted triangles
            new_tv, new_tva, new_terr, new_tdel, new_tdirty, new_tn, new_tsub = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for i in range(triangle_count):
                if not self._triangle_deleted[i]:
                    new_tv.append(self._triangle_vertex_indices[i])
                    new_tva.append(self._triangle_attribute_indices[i])
                    new_terr.append(self._triangle_edge_errors[i])
                    new_tdel.append(False)
                    new_tdirty.append(self._triangle_dirty[i])
                    new_tn.append(self._triangle_normals[i])
                    new_tsub.append(self._triangle_sub_mesh[i])
            (
                self._triangle_vertex_indices,
                self._triangle_attribute_indices,
                self._triangle_edge_errors,
            ) = (new_tv, new_tva, new_terr)
            self._triangle_deleted, self._triangle_dirty, self._triangle_normals = (
                new_tdel,
                new_tdirty,
                new_tn,
            )
            self._triangle_sub_mesh = new_tsub
            triangle_count = len(self._triangle_vertex_indices)

        self._update_references()

        if iteration == 0:
            # Reset flags
            for i in range(vertex_count):
                self._v_border[i] = False
                self._v_seam[i] = False
                self._v_foldover[i] = False

            # Find border vertices (appear in only one triangle's neighbourhood)
            border_min_x = float("inf")
            border_max_x = float("-inf")
            border_vertex_count = 0

            for i in range(vertex_count):
                tstart = self._vertex_ref_start[i]
                tcount = self._vertex_ref_count[i]
                seen_ids: dict[int, int] = {}
                for j in range(tcount):
                    tid = self._ref_triangle_id[tstart + j]
                    for k in range(3):
                        vid = int(self._triangle_vertex_indices[tid][k])
                        seen_ids[vid] = seen_ids.get(vid, 0) + 1

                for vid, cnt in seen_ids.items():
                    if cnt == 1:
                        self._v_border[vid] = True
                        border_vertex_count += 1
                        if self.options.enable_smart_link:
                            px = self._vertex_positions[vid][0]
                            if px < border_min_x:
                                border_min_x = px
                            if px > border_max_x:
                                border_max_x = px

            # Smart link: weld near-coincident border vertices
            if self.options.enable_smart_link and border_vertex_count > 0:
                border_area_width = border_max_x - border_min_x
                if border_area_width == 0:
                    border_area_width = 1.0

                border_verts = []
                for i in range(vertex_count):
                    if self._v_border[i]:
                        h = int(
                            (
                                (
                                    (self._vertex_positions[i][0] - border_min_x)
                                    / border_area_width
                                )
                                * 2.0
                                - 1.0
                            )
                            * (2**31 - 1)
                        )
                        border_verts.append((h, i))
                border_verts.sort(key=lambda x: x[0])

                link_dist = self.options.vertex_link_distance
                link_dist_sq = link_dist * link_dist
                hash_max_dist = max(
                    int((link_dist / border_area_width) * (2**31 - 1)), 1
                )

                active_border = list(
                    border_verts
                )  # (hash, index); index=-1 means consumed
                for i in range(len(active_border)):
                    hi, my_idx = active_border[i]
                    if my_idx == -1:
                        continue
                    my_pt = self._vertex_positions[my_idx]
                    for j in range(i + 1, len(active_border)):
                        hj, other_idx = active_border[j]
                        if other_idx == -1:
                            continue
                        if (hj - hi) > hash_max_dist:
                            break
                        other_pt = self._vertex_positions[other_idx]
                        sqr_mag = float(np.dot(my_pt - other_pt, my_pt - other_pt))
                        if sqr_mag <= link_dist_sq:
                            active_border[j] = (hj, -1)
                            self._v_border[my_idx] = False
                            self._v_border[other_idx] = False
                            if self._are_uvs_same(0, my_idx, other_idx):
                                self._v_foldover[my_idx] = True
                                self._v_foldover[other_idx] = True
                            else:
                                self._v_seam[my_idx] = True
                                self._v_seam[other_idx] = True

                            # Redirect other_idx → my_idx in triangles
                            o_tstart = self._vertex_ref_start[other_idx]
                            o_tcount = self._vertex_ref_count[other_idx]
                            for k in range(o_tcount):
                                tid = self._ref_triangle_id[o_tstart + k]
                                tv = self._ref_corner_index[o_tstart + k]
                                self._triangle_vertex_indices[tid][tv] = my_idx

                self._update_references()

            # Init quadrics
            for i in range(vertex_count):
                self._vertex_quadrics[i] = SymmetricMatrix()

            for i in range(triangle_count):
                v0i, v1i, v2i = (
                    int(self._triangle_vertex_indices[i][0]),
                    int(self._triangle_vertex_indices[i][1]),
                    int(self._triangle_vertex_indices[i][2]),
                )
                p0 = self._vertex_positions[v0i]
                p1 = self._vertex_positions[v1i]
                p2 = self._vertex_positions[v2i]
                # np.cross doesn't keep np.float64 type info, it thinks it's Any
                n = cast(npt.NDArray[np.float64], np.cross(p1 - p0, p2 - p0))
                nn = np.linalg.norm(n)
                if nn > 0:
                    n /= nn
                self._triangle_normals[i] = n

                d = -float(np.dot(n, p0))
                sm = SymmetricMatrix(n[0], n[1], n[2], d)
                self._vertex_quadrics[v0i] += sm
                self._vertex_quadrics[v1i] += sm
                self._vertex_quadrics[v2i] += sm

            if self.options.preserve_surface_curvature:
                self._calculate_vertex_curvatures()
            else:
                self._vert_curvatures = None

            # Calculate per-edge errors
            for i in range(triangle_count):
                e0, _, _ = self._calculate_error_with_curvature(
                    int(self._triangle_vertex_indices[i][0]),
                    int(self._triangle_vertex_indices[i][1]),
                )
                e1, _, _ = self._calculate_error_with_curvature(
                    int(self._triangle_vertex_indices[i][1]),
                    int(self._triangle_vertex_indices[i][2]),
                )
                e2, _, _ = self._calculate_error_with_curvature(
                    int(self._triangle_vertex_indices[i][2]),
                    int(self._triangle_vertex_indices[i][0]),
                )
                self._triangle_edge_errors[i][0] = e0
                self._triangle_edge_errors[i][1] = e1
                self._triangle_edge_errors[i][2] = e2
                self._triangle_edge_errors[i][3] = min(e0, e1, e2)

    # ------------------------------------------------------------------
    def _update_references(self) -> None:
        """
        rebuilds the refs structure — a lookup table that answers the question: "Which triangles touch which vertex?"
        """
        vertex_count = len(self._vertex_positions)
        triangles_count = len(self._triangle_vertex_indices)

        for i in range(vertex_count):
            self._vertex_ref_start[i] = 0
            self._vertex_ref_count[i] = 0

        for i in range(triangles_count):
            for k in range(3):
                self._vertex_ref_count[int(self._triangle_vertex_indices[i][k])] += 1

        tstart = 0
        self._remaining_vertices = 0
        for i in range(vertex_count):
            self._vertex_ref_start[i] = tstart
            if self._vertex_ref_count[i] > 0:
                tstart += self._vertex_ref_count[i]
                self._vertex_ref_count[i] = 0
                self._remaining_vertices += 1

        # Allocate ref arrays
        self._ref_triangle_id = [0] * tstart
        self._ref_corner_index = [0] * tstart

        for i in range(triangles_count):
            for k in range(3):
                vid = int(self._triangle_vertex_indices[i][k])
                pos = self._vertex_ref_start[vid] + self._vertex_ref_count[vid]
                self._ref_triangle_id[pos] = i
                self._ref_corner_index[pos] = k
                self._vertex_ref_count[vid] += 1

    # ------------------------------------------------------------------
    def _compact_mesh(self) -> None:
        vertex_count = len(self._vertex_positions)
        triangles_count = len(self._triangle_vertex_indices)

        for i in range(vertex_count):
            self._vertex_ref_count[i] = 0

        dst = 0
        new_tv, new_tva, new_terr, new_tdel, new_tdirty, new_tn, new_tsub = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(triangles_count):
            if self._triangle_deleted[i]:
                continue
            tri_v = self._triangle_vertex_indices[i].copy()
            tri_va = self._triangle_attribute_indices[i].copy()

            for slot in range(3):
                if tri_va[slot] != tri_v[slot]:
                    i_dest = int(tri_va[slot])
                    i_src = int(tri_v[slot])
                    self._vertex_positions[i_dest] = self._vertex_positions[
                        i_src
                    ].copy()
                    tri_v[slot] = tri_va[slot]

            new_tv.append(tri_v)
            new_tva.append(tri_va)
            new_terr.append(self._triangle_edge_errors[i])
            new_tdel.append(False)
            new_tdirty.append(False)
            new_tn.append(self._triangle_normals[i])
            new_tsub.append(self._triangle_sub_mesh[i])

            for k in range(3):
                self._vertex_ref_count[int(tri_v[k])] = 1

        (
            self._triangle_vertex_indices,
            self._triangle_attribute_indices,
            self._triangle_edge_errors,
        ) = (new_tv, new_tva, new_terr)
        self._triangle_deleted, self._triangle_dirty, self._triangle_normals = (
            new_tdel,
            new_tdirty,
            new_tn,
        )
        self._triangle_sub_mesh = new_tsub
        triangles_count = len(self._triangle_vertex_indices)

        # Remap vertices
        dst = 0
        new_p = []
        new_normals: list[npt.NDArray[np.float32]] | None = (
            [] if self._va_normals is not None else None
        )
        new_colors: list[npt.NDArray[np.float32]] | None = (
            [] if self._va_colors is not None else None
        )
        new_uvs: list[list[npt.NDArray[np.float32]] | None] = [
            [] if self._va_uvs[ch] is not None else None
            for ch in range(SimplificationMesh.UV_CHANNEL_COUNT)
        ]
        mapping = [-1] * vertex_count

        for i in range(vertex_count):
            if self._vertex_ref_count[i] > 0:
                mapping[i] = dst
                self._vertex_ref_start[i] = dst
                new_p.append(self._vertex_positions[i])
                if self._va_normals is not None and new_normals is not None:
                    new_normals.append(self._va_normals[i])
                if self._va_colors is not None and new_colors is not None:
                    new_colors.append(self._va_colors[i])
                for ch in range(SimplificationMesh.UV_CHANNEL_COUNT):
                    new_uvs_channel = new_uvs[ch]
                    current_uvs_channel = self._va_uvs[ch]
                    if current_uvs_channel is not None and new_uvs_channel is not None:
                        new_uvs_channel.append(current_uvs_channel[i])
                dst += 1

        # Remap triangle vertex indices
        for i in range(triangles_count):
            for k in range(3):
                old_v = int(self._triangle_vertex_indices[i][k])
                self._triangle_vertex_indices[i][k] = mapping[old_v]

        # Replace internal arrays
        self._vertex_positions = new_p
        if new_normals is not None:
            self._va_normals = new_normals
        if new_colors is not None:
            self._va_colors = new_colors
        for ch in range(SimplificationMesh.UV_CHANNEL_COUNT):
            if new_uvs[ch] is not None:
                self._va_uvs[ch] = new_uvs[ch]

        new_V = dst
        # Trim per-vertex lists to new_V
        for lst in (
            self._vertex_quadrics,
            self._vertex_ref_start,
            self._vertex_ref_count,
            self._v_border,
            self._v_seam,
            self._v_foldover,
        ):
            del lst[new_V:]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def simplify_mesh(
    mesh: SimplificationMesh,
    target_triangle_count: int,
    options: SimplificationOptions | None = None,
) -> SimplificationMesh:
    """Simplify *mesh* down to *target_triangle_count* triangles."""
    sim = FastQuadricMeshSimplification(options)
    sim.initialize(mesh)
    sim.decimate_mesh(target_triangle_count)
    return sim.to_mesh()


def simplify_mesh_lossless(
    mesh: SimplificationMesh, options: SimplificationOptions | None = None
) -> SimplificationMesh:
    """Remove degenerate / zero-error triangles without quality loss."""
    sim = FastQuadricMeshSimplification(options)
    sim.initialize(mesh)
    sim.decimate_mesh_lossless()
    return sim.to_mesh()
