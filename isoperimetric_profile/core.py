
"""Exact relative isoperimetric profile of a strictly convex polygon.

This module contains the exact branchwise formulas needed to evaluate the
relative isoperimetric profile of a polygon.  The package is intentionally
small: it keeps only the geometry and profile computation, and removes the
atlas / GPU / multi-polygon search machinery from the original code base.

Main entry points
-----------------
- ``precompute_profile``: preprocess one polygon once.
- ``profile_on_areas``: evaluate the exact profile on arbitrary areas.
- ``profile_on_uniform_grid``: evaluate on a uniform grid in area.
- ``candidate_at_area``: recover the exact minimizing candidate at one area.
- ``max_relative_ratio``: coarse/fine search in area for the maximal ratio
  ``I_P(A) / I_D(A)`` against the disk of the same total area.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PI = math.pi
TAU = 2.0 * math.pi
EPS = 1.0e-12

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------

def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Point, scalar: float) -> Point:
    return (a[0] * scalar, a[1] * scalar)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Point, b: Point) -> float:
    return norm(sub(a, b))


def unit(a: Point) -> Point:
    n = norm(a)
    if n <= EPS:
        raise ValueError("zero-length vector")
    return (a[0] / n, a[1] / n)


def perp_ccw(a: Point) -> Point:
    return (-a[1], a[0])


def perp_cw(a: Point) -> Point:
    return (a[1], -a[0])


def polygon_signed_area(polygon: Sequence[Point]) -> float:
    total = 0.0
    n = len(polygon)
    for i in range(n):
        total += cross(polygon[i], polygon[(i + 1) % n])
    return 0.5 * total


def polygon_area(polygon: Sequence[Point]) -> float:
    return abs(polygon_signed_area(polygon))


def polygon_centroid(polygon: Sequence[Point]) -> Point:
    a2 = 0.0
    cx = 0.0
    cy = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cr = x0 * y1 - y0 * x1
        a2 += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    if abs(a2) <= EPS:
        raise ValueError("degenerate polygon")
    return (cx / (3.0 * a2), cy / (3.0 * a2))


def translate(polygon: Sequence[Point], shift: Point) -> List[Point]:
    return [add(p, shift) for p in polygon]


def scale(polygon: Sequence[Point], scalar: float) -> List[Point]:
    return [mul(p, scalar) for p in polygon]


def ensure_ccw(polygon: Sequence[Point]) -> List[Point]:
    points = [(float(x), float(y)) for x, y in polygon]
    return points if polygon_signed_area(points) > 0.0 else list(reversed(points))


def remove_duplicate_endpoint(polygon: Sequence[Point], tol: float = 1.0e-12) -> List[Point]:
    points = [(float(x), float(y)) for x, y in polygon]
    if len(points) >= 2 and dist(points[0], points[-1]) <= tol:
        return points[:-1]
    return points


def remove_consecutive_collinear(polygon: Sequence[Point], tol: float = 1.0e-12) -> List[Point]:
    points = remove_duplicate_endpoint(polygon, tol=tol)
    n = len(points)
    if n < 3:
        raise ValueError("a polygon needs at least three vertices")
    out: List[Point] = []
    for i in range(n):
        a = points[(i - 1) % n]
        b = points[i]
        c = points[(i + 1) % n]
        if dist(a, b) <= tol:
            continue
        if abs(cross(sub(b, a), sub(c, b))) <= tol * max(1.0, dist(a, b), dist(b, c)):
            continue
        out.append(b)
    if len(out) < 3:
        raise ValueError("polygon collapsed after removing collinear vertices")
    return out


def validate_convex_ccw(polygon: Sequence[Point], tol: float = 1.0e-10) -> None:
    points = ensure_ccw(polygon)
    n = len(points)
    if n < 3:
        raise ValueError("a polygon needs at least three vertices")
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        c = points[(i + 2) % n]
        if cross(sub(b, a), sub(c, b)) <= tol:
            raise ValueError("polygon is not strictly convex and counterclockwise")


def prepare_polygon(polygon: Sequence[Point]) -> List[Point]:
    """Return a cleaned strictly convex CCW polygon."""
    cleaned = remove_consecutive_collinear(ensure_ccw(polygon))
    validate_convex_ccw(cleaned)
    return cleaned


def center_polygon(polygon: Sequence[Point]) -> List[Point]:
    c = polygon_centroid(polygon)
    return translate(polygon, (-c[0], -c[1]))


def scale_to_area(polygon: Sequence[Point], target_area: float) -> List[Point]:
    cleaned = prepare_polygon(polygon)
    area = polygon_area(cleaned)
    if area <= EPS:
        raise ValueError("degenerate polygon")
    factor = math.sqrt(float(target_area) / area)
    return scale(cleaned, factor)


def center_and_scale_to_area(polygon: Sequence[Point], target_area: float) -> List[Point]:
    return center_polygon(scale_to_area(polygon, target_area))


def point_segment_distance(point: Point, a: Point, b: Point) -> float:
    ab = sub(b, a)
    ap = sub(point, a)
    denom = dot(ab, ab)
    if denom <= EPS:
        return norm(ap)
    t = min(1.0, max(0.0, dot(ap, ab) / denom))
    proj = add(a, mul(ab, t))
    return dist(point, proj)


def clip_polygon_with_halfplane(
    polygon: Sequence[Point],
    normal: Point,
    level: float,
    tol: float = 1.0e-12,
) -> List[Point]:
    """Clip by the half-plane ``{x : <normal, x> <= level}``."""
    out: List[Point] = []
    nrm = unit(normal)
    points = list(polygon)
    m = len(points)

    for i in range(m):
        a = points[i]
        b = points[(i + 1) % m]
        da = dot(nrm, a) - level
        db = dot(nrm, b) - level
        inside_a = da <= tol
        inside_b = db <= tol

        if inside_a and inside_b:
            out.append(b)
        elif inside_a and not inside_b:
            t = da / (da - db)
            out.append(add(a, mul(sub(b, a), t)))
        elif (not inside_a) and inside_b:
            t = da / (da - db)
            out.append(add(a, mul(sub(b, a), t)))
            out.append(b)

    if not out:
        return []

    deduped: List[Point] = []
    for p in out:
        if not deduped or dist(p, deduped[-1]) > 1.0e-12:
            deduped.append(p)
    if len(deduped) >= 2 and dist(deduped[0], deduped[-1]) <= 1.0e-12:
        deduped.pop()
    return deduped


def chord_intersections(
    polygon: Sequence[Point],
    normal: Point,
    level: float,
    tol: float = 1.0e-10,
) -> List[Point]:
    """Intersections of ``{x : <normal, x> = level}`` with the polygon boundary."""
    nrm = unit(normal)
    points: List[Point] = []
    m = len(polygon)

    for i in range(m):
        a = polygon[i]
        b = polygon[(i + 1) % m]
        da = dot(nrm, a) - level
        db = dot(nrm, b) - level

        if abs(da) <= tol and abs(db) <= tol:
            points.extend([a, b])
            continue
        if abs(da) <= tol:
            points.append(a)
        if da * db < -tol * tol:
            t = da / (da - db)
            points.append(add(a, mul(sub(b, a), t)))
        elif abs(db) <= tol:
            points.append(b)

    unique: List[Point] = []
    for p in points:
        if all(dist(p, q) > 1.0e-8 for q in unique):
            unique.append(p)
    return unique


def chord_length(polygon: Sequence[Point], normal: Point, level: float) -> float:
    points = chord_intersections(polygon, normal, level)
    if len(points) < 2:
        return 0.0
    best = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            best = max(best, dist(points[i], points[j]))
    return best


# ---------------------------------------------------------------------------
# Exact profile objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExactCandidate:
    family: str
    free_perimeter: float
    area: float
    center: Optional[Point] = None
    radius: Optional[float] = None
    normal: Optional[Point] = None
    level: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfilePrecomputation:
    polygon: List[Point]
    total_area: float
    disk_branches: List[Dict[str, Any]]
    strip_branches: List[Dict[str, Any]]


@dataclass(frozen=True)
class ProfileValues:
    polygon: List[Point]
    total_area: float
    query_areas: np.ndarray
    reduced_areas: np.ndarray
    profile: np.ndarray
    disk_profile: np.ndarray
    ratio: np.ndarray
    families: List[str]
    max_index: int

    @property
    def max_area(self) -> float:
        return float(self.query_areas[self.max_index])

    @property
    def max_reduced_area(self) -> float:
        return float(self.reduced_areas[self.max_index])

    @property
    def max_ratio(self) -> float:
        return float(self.ratio[self.max_index])


@dataclass(frozen=True)
class RelativeProfileMaximum:
    polygon: List[Point]
    total_area: float
    best_area: float
    best_reduced_area: float
    best_ratio: float
    best_candidate: ExactCandidate
    grid: ProfileValues


# ---------------------------------------------------------------------------
# Branch precomputation
# ---------------------------------------------------------------------------

def standard_edge_geometry(polygon: Sequence[Point]) -> Dict[str, Any]:
    k = len(polygon)
    tangents: List[Point] = []
    normals: List[Point] = []
    supports: List[float] = []
    lengths: List[float] = []

    for i in range(k):
        d = sub(polygon[(i + 1) % k], polygon[i])
        t = unit(d)
        n = perp_cw(t)
        tangents.append(t)
        normals.append(n)
        supports.append(dot(n, polygon[i]))
        lengths.append(norm(d))

    return {
        "tangents": tangents,
        "normals": normals,
        "supports": supports,
        "edge_lengths": lengths,
    }


def chain_vertices(k: int, i: int, j: int) -> List[int]:
    out: List[int] = []
    t = (i + 1) % k
    while True:
        out.append(t)
        if t == j:
            break
        t = (t + 1) % k
    return out


def internal_chain_edges(k: int, i: int, j: int) -> List[int]:
    out: List[int] = []
    t = (i + 1) % k
    while t != j:
        out.append(t)
        t = (t + 1) % k
    return out


def complementary_edges(k: int, i: int, j: int) -> List[int]:
    out: List[int] = []
    t = (j + 1) % k
    while t != i:
        out.append(t)
        t = (t + 1) % k
    return out


def angle_ccw(u: Point, v: Point) -> float:
    theta = math.atan2(cross(u, v), dot(u, v))
    return theta + TAU if theta < 0.0 else theta


def disk_branch_data(polygon: Sequence[Point]) -> List[Dict[str, Any]]:
    """Exact disk branches indexed by ordered support-edge pairs."""
    poly = prepare_polygon(polygon)
    k = len(poly)
    geom = standard_edge_geometry(poly)
    tangents = geom["tangents"]
    normals = geom["normals"]
    supports = geom["supports"]
    edge_cross = [cross(poly[idx], poly[(idx + 1) % k]) for idx in range(k)]

    out: List[Dict[str, Any]] = []
    total_area = polygon_area(poly)

    for i in range(k):
        for j in range(k):
            if i == j:
                continue

            den = cross(normals[i], normals[j])
            if abs(den) <= 1.0e-12:
                continue

            center = (
                (supports[i] * perp_cw(normals[j])[0] - supports[j] * perp_cw(normals[i])[0]) / den,
                (supports[i] * perp_cw(normals[j])[1] - supports[j] * perp_cw(normals[i])[1]) / den,
            )

            ti = tangents[i]
            tj = tangents[j]
            phi = angle_ccw(tj, (-ti[0], -ti[1]))
            if phi <= 1.0e-12 or phi >= TAU - 1.0e-12:
                continue

            a_i = dot(sub(poly[i], center), ti)
            b_i = dot(sub(poly[(i + 1) % k], center), ti)
            r_start_lo = max(0.0, -b_i)
            r_start_hi = -a_i

            a_j = dot(sub(poly[j], center), tj)
            b_j = dot(sub(poly[(j + 1) % k], center), tj)
            r_end_lo = max(0.0, a_j)
            r_end_hi = b_j

            chain_vs = chain_vertices(k, i, j)
            r_vertex = max(dist(center, poly[idx]) for idx in chain_vs)

            comp_edges = complementary_edges(k, i, j)
            if comp_edges:
                r_comp = min(
                    point_segment_distance(center, poly[e], poly[(e + 1) % k])
                    for e in comp_edges
                )
            else:
                r_comp = float("inf")

            r_lo = max(r_start_lo, r_end_lo, r_vertex)
            r_hi = min(r_start_hi, r_end_hi, r_comp)
            if not (r_hi > r_lo + 1.0e-10):
                continue

            p_start = add(center, mul(ti, -r_lo))
            p_end = add(center, mul(tj, +r_lo))
            sum_internal = sum(edge_cross[e] for e in internal_chain_edges(k, i, j))
            poly_area_lo = 0.5 * (
                cross(p_start, poly[(i + 1) % k])
                + sum_internal
                + cross(poly[j], p_end)
                + cross(p_end, p_start)
            )
            sin_phi = cross(tj, (-ti[0], -ti[1]))
            a_lo = poly_area_lo + 0.5 * r_lo * r_lo * (phi - sin_phi)
            a_lo = max(0.0, min(a_lo, total_area))
            a_hi = a_lo + 0.5 * phi * (r_hi * r_hi - r_lo * r_lo)
            a_hi = max(a_lo, min(a_hi, total_area))

            theta0 = math.atan2(tj[1], tj[0]) % TAU
            theta1 = math.atan2(-ti[1], -ti[0]) % TAU

            out.append(
                {
                    "family": "disk",
                    "start_edge": i,
                    "end_edge": j,
                    "center": center,
                    "phi": phi,
                    "r_lo": r_lo,
                    "r_hi": r_hi,
                    "A_lo": a_lo,
                    "A_hi": a_hi,
                    "theta0": theta0,
                    "theta1": theta1,
                    "support_pair": (i, j),
                }
            )

    return out


def parallel_edge_classes_unoriented(
    polygon: Sequence[Point],
    tol: float = 1.0e-10,
) -> List[Tuple[Point, List[int]]]:
    def canonical_direction(direction: Point) -> Point:
        u = unit(direction)
        if u[0] < -tol or (abs(u[0]) <= tol and u[1] < 0.0):
            return (-u[0], -u[1])
        return u

    directions = [
        canonical_direction(sub(polygon[(i + 1) % len(polygon)], polygon[i]))
        for i in range(len(polygon))
    ]
    classes: List[Tuple[Point, List[int]]] = []
    used = [False] * len(polygon)

    for i, direction in enumerate(directions):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(directions)):
            if used[j]:
                continue
            if abs(cross(direction, directions[j])) <= tol:
                used[j] = True
                group.append(j)
        classes.append((direction, group))

    return classes


def unique_sorted(values: Sequence[float], tol: float = 1.0e-12) -> List[float]:
    out: List[float] = []
    for x in sorted(float(v) for v in values):
        if not out or abs(x - out[-1]) > tol:
            out.append(x)
    return out


def strip_branch_data(polygon: Sequence[Point]) -> List[Dict[str, Any]]:
    """Exact strip branches on every exact parallel-edge class."""
    poly = prepare_polygon(polygon)
    total_area = polygon_area(poly)
    out: List[Dict[str, Any]] = []

    for direction, edge_class in parallel_edge_classes_unoriented(poly):
        if len(edge_class) < 2:
            continue

        t = unit(direction)
        u = perp_ccw(t)
        s_vert = [dot(t, v) for v in poly]
        y_vert = [dot(u, v) for v in poly]
        breaks = unique_sorted(s_vert)
        if len(breaks) < 2:
            continue

        for s_lo, s_hi in zip(breaks[:-1], breaks[1:]):
            if s_hi <= s_lo + 1.0e-12:
                continue
            s_mid = 0.5 * (s_lo + s_hi)
            active: List[Tuple[int, float, float, float]] = []

            for e in range(len(poly)):
                sa = s_vert[e]
                sb = s_vert[(e + 1) % len(poly)]
                if (sa - s_mid) * (sb - s_mid) >= 0.0:
                    continue
                ya = y_vert[e]
                yb = y_vert[(e + 1) % len(poly)]
                alpha_e = (yb - ya) / (sb - sa)
                beta_e = ya - alpha_e * sa
                y_mid = alpha_e * s_mid + beta_e
                active.append((e, alpha_e, beta_e, y_mid))

            if len(active) != 2:
                continue

            active.sort(key=lambda item: item[3])
            e_lo, a_lo_e, b_lo_e, _ = active[0]
            e_hi, a_hi_e, b_hi_e, _ = active[1]

            if (e_lo not in edge_class) or (e_hi not in edge_class):
                continue

            alpha = a_hi_e - a_lo_e
            beta = b_hi_e - b_lo_e
            l_left = alpha * s_lo + beta
            l_right = alpha * s_hi + beta
            if max(l_left, l_right) <= 1.0e-12:
                continue

            clipped = clip_polygon_with_halfplane(poly, t, s_lo)
            a_left = polygon_area(clipped) if len(clipped) >= 3 else 0.0
            a_right = a_left + l_left * (s_hi - s_lo) + 0.5 * alpha * (s_hi - s_lo) * (s_hi - s_lo)
            a_left = max(0.0, min(a_left, total_area))
            a_right = max(a_left, min(a_right, total_area))
            if a_right <= a_left + 1.0e-12:
                continue

            out.append(
                {
                    "family": "strip",
                    "direction": t,
                    "edge_class": list(edge_class),
                    "edge_pair": (e_lo, e_hi),
                    "s_lo": s_lo,
                    "s_hi": s_hi,
                    "A_lo": a_left,
                    "A_hi": a_right,
                    "L_lo": l_left,
                    "alpha": alpha,
                }
            )

    return out


def precompute_profile(polygon: Sequence[Point]) -> ProfilePrecomputation:
    poly = prepare_polygon(polygon)
    return ProfilePrecomputation(
        polygon=poly,
        total_area=polygon_area(poly),
        disk_branches=disk_branch_data(poly),
        strip_branches=strip_branch_data(poly),
    )


# ---------------------------------------------------------------------------
# Disk benchmark profile
# ---------------------------------------------------------------------------

def unit_disk_cap_area_numpy(beta: np.ndarray) -> np.ndarray:
    t = np.tan(beta)
    return beta + 0.5 * t * t * (PI - 2.0 * beta) - t


def unit_disk_profile_numpy(area: np.ndarray, iterations: int = 70) -> np.ndarray:
    area = np.asarray(area, dtype=float)
    lo = np.full_like(area, 1.0e-12, dtype=float)
    hi = np.full_like(area, 0.5 * PI - 1.0e-12, dtype=float)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        area_mid = unit_disk_cap_area_numpy(mid)
        lo = np.where(area_mid < area, mid, lo)
        hi = np.where(area_mid >= area, mid, hi)
    beta = 0.5 * (lo + hi)
    return np.tan(beta) * (PI - 2.0 * beta)


def unit_disk_profile_scalar(area: float, iterations: int = 70) -> float:
    return float(unit_disk_profile_numpy(np.array([float(area)]), iterations=iterations)[0])


def disk_profile(total_area: float, areas: np.ndarray) -> np.ndarray:
    """Profile of the disk with total area ``total_area``."""
    total_area = float(total_area)
    if total_area <= 0.0:
        raise ValueError("total_area must be positive")
    areas = np.asarray(areas, dtype=float)
    radius = math.sqrt(total_area / PI)
    unit_areas = areas * (PI / total_area)
    return radius * unit_disk_profile_numpy(unit_areas)


def disk_profile_scalar(total_area: float, area: float) -> float:
    return float(disk_profile(total_area, np.array([float(area)]))[0])


def default_area_epsilon(total_area: float) -> float:
    return 2.0e-3 * float(total_area) / PI


def _reduce_areas(total_area: float, areas: np.ndarray) -> np.ndarray:
    reduced = np.minimum(areas, total_area - areas)
    if np.any(reduced <= 0.0) or np.any(reduced >= 0.5 * total_area + 1.0e-12):
        raise ValueError("areas must lie strictly between 0 and total_area")
    return reduced


# ---------------------------------------------------------------------------
# Exact profile evaluation
# ---------------------------------------------------------------------------

def _profile_from_disk_branches(pre: ProfilePrecomputation, reduced_areas: np.ndarray) -> np.ndarray:
    best = np.full_like(reduced_areas, np.inf, dtype=float)
    if not pre.disk_branches:
        return best

    phi = np.asarray([branch["phi"] for branch in pre.disk_branches], dtype=float)[:, None]
    r_lo = np.asarray([branch["r_lo"] for branch in pre.disk_branches], dtype=float)[:, None]
    a_lo = np.asarray([branch["A_lo"] for branch in pre.disk_branches], dtype=float)[:, None]
    a_hi = np.asarray([branch["A_hi"] for branch in pre.disk_branches], dtype=float)[:, None]
    aa = reduced_areas[None, :]

    valid = (aa >= a_lo - 1.0e-12) & (aa <= a_hi + 1.0e-12)
    r_sq = r_lo * r_lo + 2.0 * (aa - a_lo) / phi
    lengths = phi * np.sqrt(np.maximum(r_sq, 0.0))
    lengths = np.where(valid, lengths, np.inf)
    return np.min(lengths, axis=0)


def _profile_from_strip_branches(pre: ProfilePrecomputation, reduced_areas: np.ndarray) -> np.ndarray:
    best = np.full_like(reduced_areas, np.inf, dtype=float)
    if not pre.strip_branches:
        return best

    a_lo = np.asarray([branch["A_lo"] for branch in pre.strip_branches], dtype=float)[:, None]
    a_hi = np.asarray([branch["A_hi"] for branch in pre.strip_branches], dtype=float)[:, None]
    l_lo = np.asarray([branch["L_lo"] for branch in pre.strip_branches], dtype=float)[:, None]
    alpha = np.asarray([branch["alpha"] for branch in pre.strip_branches], dtype=float)[:, None]
    aa = reduced_areas[None, :]

    valid = (aa >= a_lo - 1.0e-12) & (aa <= a_hi + 1.0e-12)
    disc = np.maximum(l_lo * l_lo + 2.0 * alpha * (aa - a_lo), 0.0)
    lengths = np.where(np.abs(alpha) <= 1.0e-14, l_lo, np.sqrt(disc))
    lengths = np.where(valid, lengths, np.inf)
    return np.min(lengths, axis=0)


def profile_on_areas(pre: ProfilePrecomputation, areas: Sequence[float]) -> ProfileValues:
    query_areas = np.asarray(areas, dtype=float)
    if query_areas.ndim != 1:
        raise ValueError("areas must be a one-dimensional array")
    reduced = _reduce_areas(pre.total_area, query_areas)

    best_disk = _profile_from_disk_branches(pre, reduced)
    best_strip = _profile_from_strip_branches(pre, reduced)
    profile = np.minimum(best_disk, best_strip)
    families = np.where(best_disk <= best_strip, "disk", "strip").tolist()

    disk_benchmark = disk_profile(pre.total_area, reduced)
    ratio = profile / disk_benchmark
    max_index = int(np.argmax(ratio))

    return ProfileValues(
        polygon=list(pre.polygon),
        total_area=float(pre.total_area),
        query_areas=query_areas,
        reduced_areas=reduced,
        profile=profile,
        disk_profile=disk_benchmark,
        ratio=ratio,
        families=families,
        max_index=max_index,
    )


def profile_on_uniform_grid(
    pre: ProfilePrecomputation,
    num_points: int = 256,
    area_epsilon: Optional[float] = None,
) -> ProfileValues:
    total_area = pre.total_area
    eps = default_area_epsilon(total_area) if area_epsilon is None else float(area_epsilon)
    if eps <= 0.0 or eps >= 0.5 * total_area:
        raise ValueError("area_epsilon must satisfy 0 < area_epsilon < total_area/2")
    areas = np.linspace(eps, 0.5 * total_area - eps, int(num_points), dtype=float)
    return profile_on_areas(pre, areas)


def candidate_at_area(pre: ProfilePrecomputation, area: float) -> ExactCandidate:
    """Return the exact minimizing branch at one area.

    The returned candidate always uses the reduced area
    ``min(area, total_area - area)``.
    """
    total_area = pre.total_area
    reduced_area = float(min(float(area), total_area - float(area)))
    if reduced_area <= 0.0 or reduced_area >= 0.5 * total_area + 1.0e-12:
        raise ValueError("area must lie strictly between 0 and total_area")

    best: Optional[ExactCandidate] = None

    for branch in pre.disk_branches:
        if reduced_area < branch["A_lo"] - 1.0e-12 or reduced_area > branch["A_hi"] + 1.0e-12:
            continue
        radius = math.sqrt(
            max(0.0, branch["r_lo"] * branch["r_lo"] + 2.0 * (reduced_area - branch["A_lo"]) / branch["phi"])
        )
        candidate = ExactCandidate(
            family="disk",
            free_perimeter=branch["phi"] * radius,
            area=reduced_area,
            center=branch["center"],
            radius=radius,
            metadata={
                "support_pair": branch["support_pair"],
                "arc": (branch["theta0"], branch["theta1"], branch["phi"]),
                "A_interval": (branch["A_lo"], branch["A_hi"]),
                "r_interval": (branch["r_lo"], branch["r_hi"]),
            },
        )
        if best is None or candidate.free_perimeter < best.free_perimeter - 1.0e-12:
            best = candidate

    for branch in pre.strip_branches:
        if reduced_area < branch["A_lo"] - 1.0e-12 or reduced_area > branch["A_hi"] + 1.0e-12:
            continue
        delta_a = reduced_area - branch["A_lo"]
        if abs(branch["alpha"]) <= 1.0e-14:
            free_perimeter = branch["L_lo"]
            x = delta_a / max(branch["L_lo"], 1.0e-15)
        else:
            disc = max(branch["L_lo"] * branch["L_lo"] + 2.0 * branch["alpha"] * delta_a, 0.0)
            free_perimeter = math.sqrt(disc)
            x = (-branch["L_lo"] + free_perimeter) / branch["alpha"]
        level = branch["s_lo"] + x
        candidate = ExactCandidate(
            family="strip",
            free_perimeter=free_perimeter,
            area=reduced_area,
            normal=branch["direction"],
            level=level,
            metadata={
                "parallel_edge_class": list(branch["edge_class"]),
                "edge_pair": branch["edge_pair"],
                "s_interval": (branch["s_lo"], branch["s_hi"]),
                "A_interval": (branch["A_lo"], branch["A_hi"]),
            },
        )
        if best is None or candidate.free_perimeter < best.free_perimeter - 1.0e-12:
            best = candidate

    if best is None:
        raise RuntimeError("no admissible branch found")
    return best


def max_relative_ratio(
    pre: ProfilePrecomputation,
    num_points_coarse: int = 256,
    num_points_fine: int = 1024,
    area_epsilon: Optional[float] = None,
    local_window: int = 3,
) -> RelativeProfileMaximum:
    """Coarse/fine search in area for the maximal relative ratio.

    Rigorous part:
    - each profile value on each grid is evaluated from the exact disk/strip
      branch formulas.

    Heuristic part:
    - the maximizing area is located by a coarse grid and then refined on a
      finer grid in a local window.  This is a numerical search over the exact
      evaluator, not a symbolic maximization.
    """
    coarse = profile_on_uniform_grid(pre, num_points=num_points_coarse, area_epsilon=area_epsilon)
    j = coarse.max_index
    lo_idx = max(0, j - int(local_window))
    hi_idx = min(len(coarse.query_areas) - 1, j + int(local_window))
    area_lo = float(coarse.query_areas[lo_idx])
    area_hi = float(coarse.query_areas[hi_idx])

    if area_hi <= area_lo + 1.0e-15:
        best_area = float(coarse.query_areas[j])
        fine = coarse
    else:
        fine_areas = np.linspace(area_lo, area_hi, int(num_points_fine), dtype=float)
        fine = profile_on_areas(pre, fine_areas)
        best_area = float(fine.query_areas[fine.max_index])

    best_candidate = candidate_at_area(pre, best_area)
    return RelativeProfileMaximum(
        polygon=list(pre.polygon),
        total_area=float(pre.total_area),
        best_area=best_area,
        best_reduced_area=float(min(best_area, pre.total_area - best_area)),
        best_ratio=float(fine.max_ratio),
        best_candidate=best_candidate,
        grid=fine,
    )


# ---------------------------------------------------------------------------
# Candidate reconstruction for plotting / post-processing
# ---------------------------------------------------------------------------

def disk_free_arc_points(candidate: ExactCandidate, arc_samples: int = 400) -> List[Point]:
    if candidate.family != "disk" or candidate.center is None or candidate.radius is None:
        raise ValueError("disk_free_arc_points requires a disk candidate")
    theta0, _, delta = candidate.metadata["arc"]
    center = candidate.center
    radius = candidate.radius
    out: List[Point] = []
    samples = max(2, int(arc_samples))
    for j in range(samples):
        t = j / (samples - 1)
        theta = (theta0 + t * delta) % TAU
        out.append((center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)))
    return out


def disk_region_boundary(
    candidate: ExactCandidate,
    polygon: Sequence[Point],
    arc_samples: int = 400,
) -> List[Point]:
    if candidate.family != "disk" or candidate.center is None or candidate.radius is None:
        raise ValueError("disk_region_boundary requires a disk candidate")
    if "support_pair" not in candidate.metadata:
        raise ValueError("candidate metadata lacks support_pair")

    poly = prepare_polygon(polygon)
    k = len(poly)
    i, j = candidate.metadata["support_pair"]
    geom = standard_edge_geometry(poly)
    ti = geom["tangents"][i]
    tj = geom["tangents"][j]
    center = candidate.center
    radius = candidate.radius

    p_start = add(center, mul(ti, -radius))
    p_end = add(center, mul(tj, +radius))

    boundary: List[Point] = [p_start]
    for vertex_index in chain_vertices(k, i, j):
        boundary.append(poly[vertex_index])
    boundary.append(p_end)

    arc = disk_free_arc_points(candidate, arc_samples=arc_samples)
    if len(arc) >= 2:
        boundary.extend(arc[1:-1])
    return boundary


def strip_region_polygon(candidate: ExactCandidate, polygon: Sequence[Point]) -> List[Point]:
    if candidate.family != "strip" or candidate.normal is None or candidate.level is None:
        raise ValueError("strip_region_polygon requires a strip candidate")
    return clip_polygon_with_halfplane(polygon, candidate.normal, candidate.level)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def candidate_to_dict(candidate: ExactCandidate) -> Dict[str, Any]:
    return {
        "family": candidate.family,
        "free_perimeter": float(candidate.free_perimeter),
        "area": float(candidate.area),
        "center": list(candidate.center) if candidate.center is not None else None,
        "radius": float(candidate.radius) if candidate.radius is not None else None,
        "normal": list(candidate.normal) if candidate.normal is not None else None,
        "level": float(candidate.level) if candidate.level is not None else None,
        "metadata": candidate.metadata,
    }


def profile_values_to_dict(values: ProfileValues) -> Dict[str, Any]:
    return {
        "polygon": [[float(x), float(y)] for x, y in values.polygon],
        "total_area": float(values.total_area),
        "query_areas": values.query_areas.tolist(),
        "reduced_areas": values.reduced_areas.tolist(),
        "profile": values.profile.tolist(),
        "disk_profile": values.disk_profile.tolist(),
        "ratio": values.ratio.tolist(),
        "families": list(values.families),
        "max_index": int(values.max_index),
        "max_area": float(values.max_area),
        "max_reduced_area": float(values.max_reduced_area),
        "max_ratio": float(values.max_ratio),
    }


def relative_profile_maximum_to_dict(result: RelativeProfileMaximum) -> Dict[str, Any]:
    return {
        "polygon": [[float(x), float(y)] for x, y in result.polygon],
        "total_area": float(result.total_area),
        "best_area": float(result.best_area),
        "best_reduced_area": float(result.best_reduced_area),
        "best_ratio": float(result.best_ratio),
        "best_candidate": candidate_to_dict(result.best_candidate),
        "grid": profile_values_to_dict(result.grid),
    }


def plot_isoperimetric_profile(
    values: ProfileValues,
    path: str | Path | None = None,
    show: bool = False,
    title: Optional[str] = None,
    dpi: int = 160,
) -> None:
    """Plot the sampled isoperimetric profile ``I_P(A)``."""
    if path is None and not show:
        raise ValueError("either path must be provided or show must be True")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "matplotlib is required for plotting; install it with `pip install matplotlib`"
        ) from exc

    fig, ax = plt.subplots()
    ax.plot(values.reduced_areas, values.profile, linewidth=2.0, label=r"$I_P(A)$")
    ax.set_xlabel("Area")
    ax.set_ylabel("Relative Perimeter")
    ax.set_title(title or "Polygon Isoperimetric Profile")
    ax.grid(True, alpha=0.25)
    ax.legend()

    if path is not None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def regular_polygon(
    k: int,
    area: float = PI,
    rotation: float = 0.0,
    center: Point = (0.0, 0.0),
) -> List[Point]:
    if k < 3:
        raise ValueError("k must be at least 3")
    radius = math.sqrt(2.0 * float(area) / (k * math.sin(2.0 * PI / k)))
    cx, cy = center
    return [
        (
            cx + radius * math.cos(rotation + 2.0 * PI * j / k),
            cy + radius * math.sin(rotation + 2.0 * PI * j / k),
        )
        for j in range(k)
    ]


def _smoke_test() -> Dict[str, Any]:
    polygon = regular_polygon(5, area=PI)
    pre = precompute_profile(polygon)
    grid = profile_on_uniform_grid(pre, num_points=64)
    best = max_relative_ratio(pre, num_points_coarse=64, num_points_fine=256)
    candidate = candidate_at_area(pre, grid.max_area)
    return {
        "total_area": float(pre.total_area),
        "max_ratio_grid": float(grid.max_ratio),
        "max_ratio_refined": float(best.best_ratio),
        "best_area_refined": float(best.best_area),
        "candidate_family_at_grid_max": candidate.family,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(_smoke_test(), indent=2))
