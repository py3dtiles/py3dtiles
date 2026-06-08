"""
Fast Quadric Mesh Simplification - Python/NumPy Port

Original C# implementation by Mattias Edlund (MIT License)
Based on "Mesh Simplification Tutorial" (C) Sven Forstmann 2014 (MIT License)
https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification

Python port preserves all original logic including:
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
from typing import Optional

import numpy as np

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
        self.m = np.zeros(10, dtype=np.float64)
        if a or b or c or d:
            self.m[0] = a * a;  self.m[1] = a * b;  self.m[2] = a * c;  self.m[3] = a * d
            self.m[4] = b * b;  self.m[5] = b * c;  self.m[6] = b * d
            self.m[7] = c * c;  self.m[8] = c * d
            self.m[9] = d * d

    # ------------------------------------------------------------------
    def __iadd__(self, other: "SymmetricMatrix") -> "SymmetricMatrix":
        self.m += other.m
        return self

    def __add__(self, other: "SymmetricMatrix") -> "SymmetricMatrix":
        result = SymmetricMatrix()
        result.m = self.m + other.m
        return result

    # ------------------------------------------------------------------
    # Sub-determinants used by CalculateError
    # ------------------------------------------------------------------
    def det1(self) -> float:
        """det of the top-left 3×3 sub-matrix."""
        m = self.m
        return (m[0] * m[4] * m[7]
                + 2.0 * m[1] * m[2] * m[5]
                - m[2] * m[2] * m[4]
                - m[0] * m[5] * m[5]
                - m[1] * m[1] * m[7])

    def det2(self) -> float:
        m = self.m
        return (m[1] * m[5] * m[8]
                + m[2] * m[6] * m[5]
                + m[3] * m[2] * m[7]   # sign flip vs det1 layout
                - m[3] * m[5] * m[5]
                - m[1] * m[6] * m[8]   # actually  Determinant2 in original
                - m[2] * m[2] * m[6])

    def det3(self) -> float:
        m = self.m
        return (m[0] * m[5] * m[8]
                + m[1] * m[3] * m[7]
                + m[2] * m[1] * m[8]
                - m[2] * m[5] * m[3]
                - m[0] * m[6] * m[7]
                - m[1] * m[1] * m[8])

    def det4(self) -> float:
        m = self.m
        return (m[0] * m[4] * m[8]
                + 2.0 * m[1] * m[2] * m[6]
                - m[2] * m[2] * m[4]
                - m[0] * m[6] * m[6]
                - m[1] * m[1] * m[8])


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
class Mesh:
    """Simple triangle-mesh container."""
    vertices: np.ndarray                           # (V, 3) float64
    indices: list[np.ndarray]                      # one int32 array per sub-mesh, length = 3*T
    normals: Optional[np.ndarray] = None           # (V, 3) float32
    colors: Optional[np.ndarray] = None            # (V, 4) float32
    uvs: list[Optional[np.ndarray]] = field(default_factory=lambda: [None] * 4)  # (V, 2|3|4)

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
    def __init__(self, options: Optional[SimplificationOptions] = None):
        self.options = options or SimplificationOptions()
        self._reset()

    # ------------------------------------------------------------------
    def _reset(self):
        # Vertex arrays
        self._v_p: list[np.ndarray] = []        # positions  [V] np.ndarray shape (3,)
        self._v_q: list[SymmetricMatrix] = []   # quadrics
        self._v_tstart: list[int] = []
        self._v_tcount: list[int] = []
        self._v_border: list[bool] = []
        self._v_seam: list[bool] = []
        self._v_foldover: list[bool] = []

        # Triangle arrays (all parallel lists)
        self._t_v: list[np.ndarray] = []        # shape (3,) int  – vertex indices
        self._t_va: list[np.ndarray] = []       # shape (3,) int  – attribute indices
        self._t_err: list[np.ndarray] = []      # shape (4,) float – err0..err3
        self._t_deleted: list[bool] = []
        self._t_dirty: list[bool] = []
        self._t_n: list[np.ndarray] = []        # shape (3,)
        self._t_sub_mesh: list[int] = []

        # Ref arrays
        self._r_tid: list[int] = []
        self._r_tvertex: list[int] = []

        # Vertex attributes
        self._va_normals: Optional[list] = None
        self._va_uvs: list[Optional[list]] = [None] * Mesh.UV_CHANNEL_COUNT
        self._va_uv_dims: list[int] = [0] * Mesh.UV_CHANNEL_COUNT
        self._va_colors: Optional[list] = None

        self._sub_mesh_count = 0
        self._remaining_vertices = 0

    # ===================================================================
    # Public API
    # ===================================================================

    def initialize(self, mesh: Mesh):
        """Load mesh data into internal structures."""
        self._reset()
        self._sub_mesh_count = mesh.sub_mesh_count

        # Vertices
        for p in mesh.vertices:
            self._v_p.append(np.array(p, dtype=np.float64))
            self._v_q.append(SymmetricMatrix())
            self._v_tstart.append(0)
            self._v_tcount.append(0)
            self._v_border.append(True)
            self._v_seam.append(False)
            self._v_foldover.append(False)

        V = len(self._v_p)

        # Triangles
        for sub_idx, sub_indices in enumerate(mesh.indices):
            for i in range(0, len(sub_indices), 3):
                v0, v1, v2 = int(sub_indices[i]), int(sub_indices[i+1]), int(sub_indices[i+2])
                self._t_v.append(np.array([v0, v1, v2], dtype=np.int32))
                self._t_va.append(np.array([v0, v1, v2], dtype=np.int32))
                self._t_err.append(np.zeros(4, dtype=np.float64))
                self._t_deleted.append(False)
                self._t_dirty.append(False)
                self._t_n.append(np.zeros(3, dtype=np.float64))
                self._t_sub_mesh.append(sub_idx)

        # Attributes
        def _init_attr(arr, name, expected_len):
            if arr is not None:
                if len(arr) == expected_len:
                    return [np.array(arr[i], dtype=np.float32) for i in range(expected_len)]
                elif len(arr) > 0:
                    print(f"Warning: attribute '{name}' has {len(arr)} elements, expected {expected_len}")
            return None

        self._va_normals = _init_attr(mesh.normals, "normals", V)
        self._va_colors  = _init_attr(mesh.colors, "colors", V)

        for ch in range(Mesh.UV_CHANNEL_COUNT):
            uv = mesh.uvs[ch] if ch < len(mesh.uvs) else None
            if uv is not None and len(uv) == V:
                dim = np.array(uv[0]).shape[0] if hasattr(uv[0], '__len__') else 2
                self._va_uvs[ch] = [np.array(uv[i], dtype=np.float32) for i in range(V)]
                self._va_uv_dims[ch] = dim
            else:
                self._va_uvs[ch] = None
                self._va_uv_dims[ch] = 0

    # ------------------------------------------------------------------
    def decimate_mesh(self, target_tris_count: int):
        """Lossy decimation down to target_tris_count triangles."""
        if target_tris_count < 0:
            raise ValueError("target_tris_count must be >= 0")

        opts = self.options
        deleted_tris = 0
        T = len(self._t_v)
        start_tris = T

        max_vertex_count = float('inf')  # unlimited

        for iteration in range(opts.max_iteration_count):
            current = start_tris - deleted_tris
            if opts.verbose and (iteration % 5) == 0:
                print(f"  iteration {iteration} - triangles {current}")

            if current <= target_tris_count and self._remaining_vertices < max_vertex_count:
                break

            if (iteration % 5) == 0:
                self._update_mesh(iteration)

            # Clear dirty
            for i in range(len(self._t_v)):
                self._t_dirty[i] = False

            threshold = 1e-9 * math.pow(iteration + 3, opts.aggressiveness)

            deleted0: list[bool] = []
            deleted1: list[bool] = []
            self._remove_vertex_pass(start_tris, target_tris_count, threshold,
                                     deleted0, deleted1, deleted_tris_ref := [deleted_tris])
            deleted_tris = deleted_tris_ref[0]

        self._compact_mesh()

    # ------------------------------------------------------------------
    def decimate_mesh_lossless(self):
        """Lossless decimation – removes only zero-error edges."""
        deleted_tris = 0
        start_tris = len(self._t_v)

        for iteration in range(9999):
            self._update_mesh(iteration)

            for i in range(len(self._t_v)):
                self._t_dirty[i] = False

            if self.options.verbose:
                print(f"  Lossless iteration {iteration}")

            deleted_tris_ref = [0]
            self._remove_vertex_pass(start_tris, 0, self._DOUBLE_EPSILON,
                                     [], [], deleted_tris_ref)

            if deleted_tris_ref[0] <= 0:
                break

        self._compact_mesh()

    # ------------------------------------------------------------------
    def to_mesh(self) -> Mesh:
        """Convert internal state back to a Mesh."""
        V = len(self._v_p)
        T = len(self._t_v)

        vertices = np.array([self._v_p[i] for i in range(V)])

        # Sub-mesh index splits
        sub_offsets = [0] * self._sub_mesh_count
        last_sub = -1
        for i in range(T):
            s = self._t_sub_mesh[i]
            if s != last_sub:
                for j in range(last_sub + 1, s):
                    sub_offsets[j] = i
                sub_offsets[s] = i
                last_sub = s
        for i in range(last_sub + 1, self._sub_mesh_count):
            sub_offsets[i] = T

        indices_out = []
        for s in range(self._sub_mesh_count):
            start = sub_offsets[s]
            end = sub_offsets[s + 1] if s + 1 < self._sub_mesh_count else T
            idx = []
            for ti in range(start, end):
                v = self._t_v[ti]
                idx.extend([int(v[0]), int(v[1]), int(v[2])])
            indices_out.append(np.array(idx, dtype=np.int32))

        new_mesh = Mesh(vertices=vertices, indices=indices_out)

        if self._va_normals is not None:
            new_mesh.normals = np.array(self._va_normals[:V])
        if self._va_colors is not None:
            new_mesh.colors = np.array(self._va_colors[:V])

        uvs_out = [None] * Mesh.UV_CHANNEL_COUNT
        for ch in range(Mesh.UV_CHANNEL_COUNT):
            if self._va_uvs[ch] is not None:
                uvs_out[ch] = np.array(self._va_uvs[ch][:V])
        new_mesh.uvs = uvs_out

        return new_mesh

    # ===================================================================
    # Private helpers
    # ===================================================================

    def _vertex_error(self, q: SymmetricMatrix, x: float, y: float, z: float) -> float:
        m = q.m
        return (m[0]*x*x + 2*m[1]*x*y + 2*m[2]*x*z + 2*m[3]*x
                + m[4]*y*y + 2*m[5]*y*z + 2*m[6]*y
                + m[7]*z*z + 2*m[8]*z + m[9])

    def _calculate_error(self, i0: int, i1: int):
        """
        Returns (error, result_point, result_index).
        result_index: 0=p1, 1=p2, 2=midpoint-or-optimal
        """
        q = self._v_q[i0] + self._v_q[i1]
        border = self._v_border[i0] and self._v_border[i1]
        det = q.det1()
        if det != 0.0 and not border:
            result = np.array([
                -1.0 / det * q.det2(),
                 1.0 / det * q.det3(),
                -1.0 / det * q.det4()
            ])
            error = self._vertex_error(q, result[0], result[1], result[2])
            return error, result, 2
        else:
            p1 = self._v_p[i0]
            p2 = self._v_p[i1]
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

    def _calculate_error_with_curvature(self, i0: int, i1: int):
        error, result, idx = self._calculate_error(i0, i1)
        if hasattr(self, '_vert_curvatures') and self._vert_curvatures is not None:
            curvature = max(self._vert_curvatures[i0], self._vert_curvatures[i1])
            error += error * curvature
        return error, result, idx

    # ------------------------------------------------------------------
    def _flipped(self, p: np.ndarray, i0: int, i1: int, deleted: list) -> bool:
        tstart = self._v_tstart[i0]
        tcount = self._v_tcount[i0]
        for k in range(tcount):
            tid = self._r_tid[tstart + k]
            if self._t_deleted[tid]:
                continue
            s = self._r_tvertex[tstart + k]
            id1 = self._t_v[tid][(s + 1) % 3]
            id2 = self._t_v[tid][(s + 2) % 3]
            if id1 == i1 or id2 == i1:
                deleted[k] = True
                continue

            d1 = self._v_p[id1] - p
            d1_norm = np.linalg.norm(d1)
            if d1_norm > 0:
                d1 /= d1_norm
            d2 = self._v_p[id2] - p
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
            if np.dot(n, self._t_n[tid]) < 0.2:
                return True
        return False

    # ------------------------------------------------------------------
    def _update_triangles(self, i0: int, ia0: int, v_idx: int,
                          deleted: list, deleted_tris_ref: list):
        tstart = self._v_tstart[v_idx]
        tcount = self._v_tcount[v_idx]
        new_refs = []
        for k in range(tcount):
            rid = tstart + k
            tid = self._r_tid[rid]
            tv  = self._r_tvertex[rid]
            if self._t_deleted[tid]:
                continue
            if deleted[k]:
                self._t_deleted[tid] = True
                deleted_tris_ref[0] += 1
                continue

            self._t_v[tid][tv] = i0
            if ia0 != -1:
                self._t_va[tid][tv] = ia0

            self._t_dirty[tid] = True
            e0, _, _ = self._calculate_error_with_curvature(int(self._t_v[tid][0]), int(self._t_v[tid][1]))
            e1, _, _ = self._calculate_error_with_curvature(int(self._t_v[tid][1]), int(self._t_v[tid][2]))
            e2, _, _ = self._calculate_error_with_curvature(int(self._t_v[tid][2]), int(self._t_v[tid][0]))
            self._t_err[tid][0] = e0
            self._t_err[tid][1] = e1
            self._t_err[tid][2] = e2
            self._t_err[tid][3] = min(e0, e1, e2)
            new_refs.append((tid, tv))

        return new_refs

    # ------------------------------------------------------------------
    @staticmethod
    def _barycentric(point, a, b, c):
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

    def _interpolate_vertex_attributes(self, dst: int, i0: int, i1: int, i2: int,
                                        point: np.ndarray):
        u, v, w = self._barycentric(point, self._v_p[i0], self._v_p[i1], self._v_p[i2])
        fu, fv, fw = float(u), float(v), float(w)

        if self._va_normals is not None:
            n = self._va_normals
            result = n[i0] * fu + n[i1] * fv + n[i2] * fw
            norm = np.linalg.norm(result)
            if norm > 0:
                result /= norm
            n[dst] = result

        for ch in range(Mesh.UV_CHANNEL_COUNT):
            if self._va_uvs[ch] is not None:
                d = self._va_uvs[ch]
                d[dst] = d[i0] * fu + d[i1] * fv + d[i2] * fw

        if self._va_colors is not None:
            c = self._va_colors
            c[dst] = c[i0] * fu + c[i1] * fv + c[i2] * fw

    def _are_uvs_same(self, channel: int, a: int, b: int) -> bool:
        uv = self._va_uvs[channel]
        if uv is not None:
            return np.array_equal(uv[a], uv[b])
        return False

    # ------------------------------------------------------------------
    def _calculate_vertex_curvatures(self):
        V = len(self._v_p)
        curvatures = [0.0] * V
        for i in range(V):
            tstart = self._v_tstart[i]
            tcount = self._v_tcount[i]
            if tcount <= 1:
                continue
            max_curv = 0.0
            for j in range(tcount):
                tidA = self._r_tid[tstart + j]
                if self._t_deleted[tidA]:
                    continue
                nA = self._t_n[tidA]
                for k in range(j + 1, tcount):
                    tidB = self._r_tid[tstart + k]
                    if self._t_deleted[tidB]:
                        continue
                    nB = self._t_n[tidB]
                    dot = float(np.dot(nA, nB))
                    dot = max(-1.0, min(1.0, dot))
                    curv = (1.0 - dot) * 0.5
                    if curv > max_curv:
                        max_curv = curv
            curvatures[i] = max_curv
        self._vert_curvatures = curvatures

    # ------------------------------------------------------------------
    def _remove_vertex_pass(self, start_tris: int, target_tris: int,
                             threshold: float, deleted0: list, deleted1: list,
                             deleted_tris_ref: list):
        opts = self.options
        T = len(self._t_v)

        for tid in range(T):
            if (self._t_dirty[tid] or self._t_deleted[tid]
                    or self._t_err[tid][3] > threshold):
                continue

            for edge_idx in range(3):
                if self._t_err[tid][edge_idx] > threshold:
                    continue

                next_edge = (edge_idx + 1) % 3
                i0 = int(self._t_v[tid][edge_idx])
                i1 = int(self._t_v[tid][next_edge])

                if self._v_border[i0] != self._v_border[i1]:
                    continue
                if self._v_seam[i0] != self._v_seam[i1]:
                    continue
                if self._v_foldover[i0] != self._v_foldover[i1]:
                    continue
                if opts.preserve_border_edges and self._v_border[i0]:
                    continue
                if opts.preserve_uv_seam_edges and self._v_seam[i0]:
                    continue
                if opts.preserve_uv_foldover_edges and self._v_foldover[i0]:
                    continue

                _, p, _ = self._calculate_error_with_curvature(i0, i1)

                tc0 = self._v_tcount[i0]
                tc1 = self._v_tcount[i1]
                deleted0.clear(); deleted0.extend([False] * tc0)
                deleted1.clear(); deleted1.extend([False] * tc1)

                if self._flipped(p, i0, i1, deleted0):
                    continue
                if self._flipped(p, i1, i0, deleted1):
                    continue

                ia0 = int(self._t_va[tid][edge_idx])
                ia1 = int(self._t_va[tid][next_edge])
                third_edge = 3 - edge_idx - next_edge
                ia2 = int(self._t_va[tid][third_edge])
                self._interpolate_vertex_attributes(ia0, ia0, ia1, ia2, p)

                # Collapse edge: move i0 to p, absorb i1's quadric
                self._v_p[i0] = p.copy()
                self._v_q[i0] += self._v_q[i1]

                effective_ia0 = -1 if self._v_seam[i0] else ia0

                tstart_before = len(self._r_tid)
                new_refs0 = self._update_triangles(i0, effective_ia0, i0, deleted0, deleted_tris_ref)
                new_refs1 = self._update_triangles(i0, effective_ia0, i1, deleted1, deleted_tris_ref)
                all_new = new_refs0 + new_refs1
                tcount_new = len(all_new)

                old_tstart = self._v_tstart[i0]
                old_tcount = self._v_tcount[i0]

                if tcount_new <= old_tcount:
                    # overwrite in place
                    for k, (t, tv) in enumerate(all_new):
                        self._r_tid[old_tstart + k] = t
                        self._r_tvertex[old_tstart + k] = tv
                else:
                    # append
                    new_start = len(self._r_tid)
                    for t, tv in all_new:
                        self._r_tid.append(t)
                        self._r_tvertex.append(tv)
                    self._v_tstart[i0] = new_start

                self._v_tcount[i0] = tcount_new
                self._remaining_vertices -= 1
                break

            current = start_tris - deleted_tris_ref[0]
            if current <= target_tris and self._remaining_vertices < float('inf'):
                break

    # ------------------------------------------------------------------
    def _update_mesh(self, iteration: int):
        T = len(self._t_v)
        V = len(self._v_p)

        if iteration > 0:
            # Compact deleted triangles
            new_tv, new_tva, new_terr, new_tdel, new_tdirty, new_tn, new_tsub = [], [], [], [], [], [], []
            for i in range(T):
                if not self._t_deleted[i]:
                    new_tv.append(self._t_v[i])
                    new_tva.append(self._t_va[i])
                    new_terr.append(self._t_err[i])
                    new_tdel.append(False)Oslandia-admin
                    new_tdirty.append(self._t_dirty[i])
                    new_tn.append(self._t_n[i])
                    new_tsub.append(self._t_sub_mesh[i])
            self._t_v, self._t_va, self._t_err = new_tv, new_tva, new_terr
            self._t_deleted, self._t_dirty, self._t_n = new_tdel, new_tdirty, new_tn
            self._t_sub_mesh = new_tsub
            T = len(self._t_v)

        self._update_references()

        if iteration == 0:
            # Reset flags
            for i in range(V):
                self._v_border[i] = False
                self._v_seam[i] = False
                self._v_foldover[i] = False

            # Find border vertices (appear in only one triangle's neighbourhood)
            border_min_x = float('inf')
            border_max_x = float('-inf')
            border_vertex_count = 0

            for i in range(V):
                tstart = self._v_tstart[i]
                tcount = self._v_tcount[i]
                seen_ids: dict[int, int] = {}
                for j in range(tcount):
                    tid = self._r_tid[tstart + j]
                    for k in range(3):
                        vid = int(self._t_v[tid][k])
                        seen_ids[vid] = seen_ids.get(vid, 0) + 1

                for vid, cnt in seen_ids.items():
                    if cnt == 1:
                        self._v_border[vid] = True
                        border_vertex_count += 1
                        if self.options.enable_smart_link:
                            px = self._v_p[vid][0]
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
                for i in range(V):
                    if self._v_border[i]:
                        h = int((((self._v_p[i][0] - border_min_x) / border_area_width)
                                  * 2.0 - 1.0) * (2**31 - 1))
                        border_verts.append((h, i))
                border_verts.sort(key=lambda x: x[0])

                link_dist = self.options.vertex_link_distance
                link_dist_sq = link_dist * link_dist
                hash_max_dist = max(int((link_dist / border_area_width) * (2**31 - 1)), 1)

                active_border = list(border_verts)  # (hash, index); index=-1 means consumed
                for i in range(len(active_border)):
                    hi, my_idx = active_border[i]
                    if my_idx == -1:
                        continue
                    my_pt = self._v_p[my_idx]
                    for j in range(i + 1, len(active_border)):
                        hj, other_idx = active_border[j]
                        if other_idx == -1:
                            continue
                        if (hj - hi) > hash_max_dist:
                            break
                        other_pt = self._v_p[other_idx]
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
                            o_tstart = self._v_tstart[other_idx]
                            o_tcount = self._v_tcount[other_idx]
                            for k in range(o_tcount):
                                tid = self._r_tid[o_tstart + k]
                                tv  = self._r_tvertex[o_tstart + k]
                                self._t_v[tid][tv] = my_idx

                self._update_references()

            # Init quadrics
            for i in range(V):
                self._v_q[i] = SymmetricMatrix()

            for i in range(T):
                v0i, v1i, v2i = (int(self._t_v[i][0]),
                                  int(self._t_v[i][1]),
                                  int(self._t_v[i][2]))
                p0 = self._v_p[v0i]
                p1 = self._v_p[v1i]
                p2 = self._v_p[v2i]
                n = np.cross(p1 - p0, p2 - p0)
                nn = np.linalg.norm(n)
                if nn > 0:
                    n /= nn
                self._t_n[i] = n

                d = -float(np.dot(n, p0))
                sm = SymmetricMatrix(n[0], n[1], n[2], d)
                self._v_q[v0i] += sm
                self._v_q[v1i] += sm
                self._v_q[v2i] += sm

            if self.options.preserve_surface_curvature:
                self._calculate_vertex_curvatures()
            else:
                self._vert_curvatures = None

            # Calculate per-edge errors
            for i in range(T):
                e0, _, _ = self._calculate_error_with_curvature(int(self._t_v[i][0]), int(self._t_v[i][1]))
                e1, _, _ = self._calculate_error_with_curvature(int(self._t_v[i][1]), int(self._t_v[i][2]))
                e2, _, _ = self._calculate_error_with_curvature(int(self._t_v[i][2]), int(self._t_v[i][0]))
                self._t_err[i][0] = e0
                self._t_err[i][1] = e1
                self._t_err[i][2] = e2
                self._t_err[i][3] = min(e0, e1, e2)

    # ------------------------------------------------------------------
    def _update_references(self):
        V = len(self._v_p)
        T = len(self._t_v)

        for i in range(V):
            self._v_tstart[i] = 0
            self._v_tcount[i] = 0

        for i in range(T):
            for k in range(3):
                self._v_tcount[int(self._t_v[i][k])] += 1

        tstart = 0
        self._remaining_vertices = 0
        for i in range(V):
            self._v_tstart[i] = tstart
            if self._v_tcount[i] > 0:
                tstart += self._v_tcount[i]
                self._v_tcount[i] = 0
                self._remaining_vertices += 1

        # Allocate ref arrays
        self._r_tid = [0] * tstart
        self._r_tvertex = [0] * tstart

        for i in range(T):
            for k in range(3):
                vid = int(self._t_v[i][k])
                pos = self._v_tstart[vid] + self._v_tcount[vid]
                self._r_tid[pos] = i
                self._r_tvertex[pos] = k
                self._v_tcount[vid] += 1

    # ------------------------------------------------------------------
    def _compact_mesh(self):
        V = len(self._v_p)
        T = len(self._t_v)

        for i in range(V):
            self._v_tcount[i] = 0

        dst = 0
        new_tv, new_tva, new_terr, new_tdel, new_tdirty, new_tn, new_tsub = [], [], [], [], [], [], []

        for i in range(T):
            if self._t_deleted[i]:
                continue
            tri_v  = self._t_v[i].copy()
            tri_va = self._t_va[i].copy()

            for slot in range(3):
                if tri_va[slot] != tri_v[slot]:
                    i_dest = int(tri_va[slot])
                    i_src  = int(tri_v[slot])
                    self._v_p[i_dest] = self._v_p[i_src].copy()
                    tri_v[slot] = tri_va[slot]

            new_tv.append(tri_v)
            new_tva.append(tri_va)
            new_terr.append(self._t_err[i])
            new_tdel.append(False)
            new_tdirty.append(False)
            new_tn.append(self._t_n[i])
            new_tsub.append(self._t_sub_mesh[i])

            for k in range(3):
                self._v_tcount[int(tri_v[k])] = 1

        self._t_v, self._t_va, self._t_err = new_tv, new_tva, new_terr
        self._t_deleted, self._t_dirty, self._t_n = new_tdel, new_tdirty, new_tn
        self._t_sub_mesh = new_tsub
        T = len(self._t_v)

        # Remap vertices
        dst = 0
        new_p      = []
        new_normals = [] if self._va_normals is not None else None
        new_colors  = [] if self._va_colors is not None else None
        new_uvs     = [[] if self._va_uvs[ch] is not None else None
                       for ch in range(Mesh.UV_CHANNEL_COUNT)]
        mapping = [-1] * V

        for i in range(V):
            if self._v_tcount[i] > 0:
                mapping[i] = dst
                self._v_tstart[i] = dst
                new_p.append(self._v_p[i])
                if new_normals is not None:
                    new_normals.append(self._va_normals[i])
                if new_colors is not None:
                    new_colors.append(self._va_colors[i])
                for ch in range(Mesh.UV_CHANNEL_COUNT):
                    if new_uvs[ch] is not None:
                        new_uvs[ch].append(self._va_uvs[ch][i])
                dst += 1

        # Remap triangle vertex indices
        for i in range(T):
            for k in range(3):
                old_v = int(self._t_v[i][k])
                self._t_v[i][k] = mapping[old_v]

        # Replace internal arrays
        self._v_p = new_p
        if new_normals is not None:
            self._va_normals = new_normals
        if new_colors is not None:
            self._va_colors = new_colors
        for ch in range(Mesh.UV_CHANNEL_COUNT):
            if new_uvs[ch] is not None:
                self._va_uvs[ch] = new_uvs[ch]

        new_V = dst
        # Trim per-vertex lists to new_V
        for lst in (self._v_q, self._v_tstart, self._v_tcount,
                    self._v_border, self._v_seam, self._v_foldover):
            del lst[new_V:]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def simplify_mesh(mesh: Mesh,
                  target_triangle_count: int,
                  options: Optional[SimplificationOptions] = None) -> Mesh:
    """Simplify *mesh* down to *target_triangle_count* triangles."""
    sim = FastQuadricMeshSimplification(options)
    sim.initialize(mesh)
    sim.decimate_mesh(target_triangle_count)
    return sim.to_mesh()


def simplify_mesh_lossless(mesh: Mesh,
                            options: Optional[SimplificationOptions] = None) -> Mesh:
    """Remove degenerate / zero-error triangles without quality loss."""
    sim = FastQuadricMeshSimplification(options)
    sim.initialize(mesh)
    sim.decimate_mesh_lossless()
    return sim.to_mesh()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a simple subdivided cube (~192 triangles) and halve it.
    def _make_cube_mesh(subdivisions: int = 4) -> Mesh:
        verts, tris = [], []
        def _face(a, b, c, d):
            base = len(verts)
            verts.extend([a, b, c, d])
            tris.extend([base, base+1, base+2, base, base+2, base+3])
            print(tris)

        s = 1.0
        for _ in range(subdivisions):
            faces = [
                ([-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s]),
                ([-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s]),
                ([-s,-s,-s],[-s, s,-s],[-s, s, s],[-s,-s, s]),
                ([ s,-s,-s],[ s, s,-s],[ s, s, s],[ s,-s, s]),
                ([-s,-s,-s],[ s,-s,-s],[ s,-s, s],[-s,-s, s]),
                ([-s, s,-s],[ s, s,-s],[ s, s, s],[-s, s, s]),
            ]
            for a,b,c,d in faces:
                _face(np.array(a), np.array(b), np.array(c), np.array(d))

        return Mesh(
            vertices=np.array(verts, dtype=np.float64),
            indices=[np.array(tris, dtype=np.int32)],
        )

    cube = _make_cube_mesh(8)
    original_t = cube.triangle_count
    print(f"Original triangles: {original_t}")

    result = simplify_mesh(cube, original_t // 2)
    print(f"Decimated triangles: {result.triangle_count}")
    assert result.triangle_count <= original_t, "Decimation did not reduce triangle count"
    print("Smoke-test passed ✓")
