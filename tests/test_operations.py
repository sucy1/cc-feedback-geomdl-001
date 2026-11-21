"""Tests for the NURBS-Python package.

Requires "pytest" to run.
"""

import math

from pytest import fixture, raises

from geomdl import BSpline, convert, fitting, operations
from geomdl.operations import offset_surface, surface_area
from geomdl.exceptions import GeomdlException

GEOMDL_DELTA = 0.001


@fixture
def plane_surf():
    """Planar surface on z=0 with u in [0, 2], v in [0, 3] (area = 6)."""
    surf = BSpline.Surface()
    surf.degree_u = 1
    surf.degree_v = 1
    surf.ctrlpts_size_u = 2
    surf.ctrlpts_size_v = 2
    surf.ctrlpts = [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]]
    surf.knotvector_u = [0.0, 0.0, 2.0, 2.0]
    surf.knotvector_v = [0.0, 0.0, 3.0, 3.0]
    return surf


def make_cylinder(radius, height, num_segs):
    """B-spline cylinder of the given radius/height, closed (periodic) in the u-direction."""
    pts = []
    for i in range(num_segs + 1):
        t = 2.0 * math.pi * i / num_segs
        x, y = radius * math.cos(t), radius * math.sin(t)
        pts.append([x, y, 0.0])
        pts.append([x, y, height])
    return fitting.interpolate_surface(pts, num_segs + 1, 2, 3, 1)


@fixture
def cylinder_surf():
    """Cylinder with r=2, h=5, closed in u."""
    return make_cylinder(2.0, 5.0, 36)


@fixture
def sphere_surf():
    """B-spline approximation of the r=1 sphere (poles excluded) with outward normals."""
    num_u, num_v = 10, 20
    pts = []
    for i in range(num_u):
        theta = math.pi * (i + 0.5) / num_u
        for j in range(num_v):
            phi = 2.0 * math.pi * j / num_v
            pts.append(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ]
            )
    surf = fitting.interpolate_surface(pts, num_u, num_v, 3, 3)
    # Sanity: normals must point outward for the negative-offset test to be meaningful
    pt, nv = operations.normal(surf, [0.5, 0.5])
    assert sum(a * b for a, b in zip(pt, nv)) > 0.0
    return surf


@fixture
def freeform_surf():
    """Smooth, well-resolved freeform surface (13x13 control net)."""
    n = 13
    pts = []
    for i in range(n):
        x = 6.0 * i / (n - 1)
        for j in range(n):
            y = 6.0 * j / (n - 1)
            pts.append([x, y, 0.3 * math.sin(0.5 * x) * math.cos(0.5 * y)])
    return fitting.interpolate_surface(pts, n, n, 3, 3)


def test_exported():
    assert "offset_surface" in dir(operations)
    assert "surface_area" in dir(operations)
    assert callable(offset_surface)
    assert callable(surface_area)


def test_surface_area_plane(plane_surf):
    area = surface_area(plane_surf)
    assert abs(area - 6.0) / 6.0 < 1e-9


def test_surface_area_plane_sample_pair(plane_surf):
    area = surface_area(plane_surf, sample_size=(21, 30))
    assert abs(area - 6.0) / 6.0 < 1e-9


def test_surface_area_cylinder(cylinder_surf):
    # Lateral area of the r=2, h=5 cylinder; u-direction is closed
    exact = 2.0 * math.pi * 2.0 * 5.0
    area = surface_area(cylinder_surf)
    assert abs(area - exact) / exact < 1e-3


def test_surface_area_invalid_sample_size(plane_surf):
    with raises(GeomdlException):
        surface_area(plane_surf, sample_size=-5)
    with raises(GeomdlException):
        surface_area(plane_surf, sample_size=(10, -1))


def test_offset_surface_plane_z(plane_surf):
    off = offset_surface(plane_surf, 2.0)
    for i in range(6):
        for j in range(6):
            u, v = i / 5.0, j / 5.0
            pt = off.evaluate_single((u, v))
            assert abs(pt[2] - 2.0) < 1e-6


def test_offset_surface_cylinder_radius(cylinder_surf):
    off = offset_surface(cylinder_surf, 1.0)
    for i in range(10):
        for j in range(5):
            u, v = i / 10.0, j / 4.0
            pt = off.evaluate_single((u, v))
            radius = math.hypot(pt[0], pt[1])
            assert abs(radius - 3.0) < 1e-3
            assert -1e-6 <= pt[2] <= 5.0 + 1e-6


def test_offset_surface_sphere_excessive_negative(sphere_surf):
    # Offsetting the r=1 sphere inwards by more than its radius of curvature must not
    # produce a self-intersecting surface
    with raises(GeomdlException, match="offset exceeds local curvature radius"):
        offset_surface(sphere_surf, -1.5)


def test_offset_surface_degrees_and_knotvector(freeform_surf):
    off = offset_surface(freeform_surf, 0.25)

    # Degrees and control net size follow the source surface by default
    assert off.degree_u == freeform_surf.degree_u
    assert off.degree_v == freeform_surf.degree_v
    assert off.ctrlpts_size_u == freeform_surf.ctrlpts_size_u
    assert off.ctrlpts_size_v == freeform_surf.ctrlpts_size_v

    # Knot vectors: correct length, non-decreasing, clamped ends (multiplicity degree + 1)
    for kv, nc, deg in (
        (off.knotvector_u, off.ctrlpts_size_u, off.degree_u),
        (off.knotvector_v, off.ctrlpts_size_v, off.degree_v),
    ):
        assert len(kv) == nc + deg + 1
        assert all(kv[i] <= kv[i + 1] for i in range(len(kv) - 1))
        assert all(k == kv[0] for k in kv[: deg + 1])
        assert all(k == kv[-1] for k in kv[-(deg + 1) :])
        # Interior knot multiplicity may not exceed the degree
        interior = kv[deg + 1 : -(deg + 1)]
        for k in set(interior):
            assert interior.count(k) <= deg


def test_offset_surface_custom_degree_and_ctrlpts(plane_surf):
    off = offset_surface(plane_surf, 1.0, degree=(2, 3), num_ctrlpts=(5, 6))
    assert off.degree_u == 2
    assert off.degree_v == 3
    assert off.ctrlpts_size_u == 5
    assert off.ctrlpts_size_v == 6
    pt = off.evaluate_single((0.5, 0.5))
    assert abs(pt[2] - 1.0) < 1e-6


def test_offset_surface_closure_preserved(cylinder_surf):
    off = offset_surface(cylinder_surf, 0.5)
    size_u = off.ctrlpts_size_u
    size_v = off.ctrlpts_size_v
    # The u-direction of the source is closed: first and last control point rows must coincide
    for j in range(size_v):
        first = off.ctrlpts[j]
        last = off.ctrlpts[(size_u - 1) * size_v + j]
        assert all(abs(a - b) < 1e-9 for a, b in zip(first, last))


def test_offset_surface_residual_smooth(freeform_surf, cylinder_surf):
    # Least-squares residual at the sampled parameters must stay below 1e-4 for smooth surfaces
    def max_residual(src, off, dist, size_u, size_v, closed_u):
        err = 0.0
        for i in range(size_u + (0 if closed_u else 1)):
            u = i / size_u
            for j in range(size_v + 1):
                v = j / size_v
                fit_pt = off.evaluate_single((u, v))
                pt, nv = operations.normal(src, [u, v])
                target = [c + dist * n for c, n in zip(pt, nv)]
                err = max(err, math.dist(fit_pt, target))
        return err

    # Open smooth surface, explicit sample grid
    off1 = offset_surface(freeform_surf, 0.3, sample_size=(27, 27))
    assert max_residual(freeform_surf, off1, 0.3, 27, 27, False) < 1e-4

    # Closed (periodic) u-direction, sampled half-open
    cyl = make_cylinder(2.0, 5.0, 72)
    off2 = offset_surface(cyl, 1.0, sample_size=(150, 8))
    assert max_residual(cyl, off2, 1.0, 150, 7, True) < 1e-4


def test_offset_surface_rational_rejected(plane_surf):
    nurbs_surf = convert.bspline_to_nurbs(plane_surf)
    assert nurbs_surf.rational
    with raises(GeomdlException, match="rational offset not supported"):
        offset_surface(nurbs_surf, 1.0)


def test_offset_surface_invalid_arguments(plane_surf):
    # Negative sample sizes
    with raises(GeomdlException):
        offset_surface(plane_surf, 1.0, sample_size=-3)
    with raises(GeomdlException):
        offset_surface(plane_surf, 1.0, sample_size=(10, -2))
    # Degrees below 1
    with raises(GeomdlException):
        offset_surface(plane_surf, 1.0, degree=0)
    with raises(GeomdlException):
        offset_surface(plane_surf, 1.0, degree=(2, 0))
    # Fewer control points than degree + 1
    with raises(GeomdlException):
        offset_surface(plane_surf, 1.0, degree=3, num_ctrlpts=(3, 5))
    with raises(GeomdlException):
        offset_surface(plane_surf, 1.0, num_ctrlpts=1)
