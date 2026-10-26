"""v0.1 spline/STEP helpers. Requires a compatible OpenCASCADE Python runtime; no full sewing pipeline, tolerance policy or unit guarantee yet."""

import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.collections import Array1_double, Array1_gp_Pnt2d, Array1_int
from OCP.Geom2d import Geom2d_BSplineCurve
from OCP.gp import gp_Pnt2d
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from scipy.interpolate import make_interp_spline


def knots_ocp(vals):
    k, m = np.unique(vals, return_counts=True)
    a = Array1_double(1, len(k))
    b = Array1_int(1, len(k))
    for i, (x, y) in enumerate(zip(k, m)):
        a.SetValue(i + 1, float(x))
        b.SetValue(i + 1, int(y))
    return a, b


def make_2d_bspline_edge(uv_pts, surf, deg=3):
    uv_pts = np.asarray(uv_pts, dtype=float)
    _, idx = np.unique(np.round(uv_pts, 12), axis=0, return_index=True)
    uv_pts = uv_pts[np.sort(idx)]
    t = np.linspace(0.0, 1.0, len(uv_pts))
    k = min(deg, len(uv_pts) - 1)
    sp = make_interp_spline(t, uv_pts, k=k)
    c = sp.c
    arr = Array1_gp_Pnt2d(1, len(c))
    for i, pt in enumerate(c):
        arr.SetValue(i + 1, gp_Pnt2d(float(pt[0]), float(pt[1])))
    uk, um = knots_ocp(sp.t)
    c2d = Geom2d_BSplineCurve(arr, uk, um, k)
    return BRepBuilderAPI_MakeEdge(c2d, surf, 0.0, 1.0).Edge()


def read_step_shape(path):
    r = STEPControl_Reader()
    assert r.ReadFile(str(path)) == IFSelect_RetDone, f"Failed to read {path}"
    r.TransferRoots()
    return r.OneShape()


def write_step_checked(shape, path):
    assert BRepCheck_Analyzer(shape).IsValid(), f"Invalid shape before STEP write: {path}"
    w = STEPControl_Writer()
    assert w.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert w.Write(str(path)) == IFSelect_RetDone
    rt = read_step_shape(path)
    assert BRepCheck_Analyzer(rt).IsValid(), f"Invalid shape after STEP round-trip: {path}"
    return rt
