"""Prescribed transition primitives from rejected ruled/tangent trials."""

import numpy as np

from ..cad_closure_20261028.cad_common import items


def blend_endpoints(original_targets, first, last):
    """Smoothstep endpoint correction, before the original every-fourth downsample."""
    parameter = np.linspace(0, 1, len(original_targets))
    weight = 3 * parameter * parameter - 2 * parameter * parameter * parameter
    return (
        original_targets
        + (1 - weight[:, None]) * (first - original_targets[0])
        + weight[:, None] * (last - original_targets[-1])
    )


def ruled(edge, target_points):
    """Caller supplies prescribed targets; no specimen-based target inference."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepFill import BRepFill
    from OCP.collections import HArray1_double, HArray1_gp_Pnt
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.gp import gp_Pnt

    points = target_points[::4]
    array = HArray1_gp_Pnt(1, len(points))
    for index, point in enumerate(points, 1):
        array.SetValue(index, gp_Pnt(*point))
    parameters = HArray1_double(1, len(points))
    for index, parameter in enumerate(np.linspace(0, 1, len(points)), 1):
        parameters.SetValue(index, float(parameter))
    fit = GeomAPI_Interpolate(array, parameters, False, 1e-8)
    fit.Perform()
    assert fit.IsDone()
    target = BRepBuilderAPI_MakeEdge(fit.Curve()).Edge()
    target.Orientation(edge.Orientation())
    return BRepFill.Face_s(edge, target)


def tangent(edge, support_face, ruled_face):
    """Initial G1 fill trial; no validity, collision or successful reload claim."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCP.GeomAbs import GeomAbs_C0, GeomAbs_G1
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS

    fill = BRepOffsetAPI_MakeFilling(3, 30, 3, False, 1e-6, 1e-5, 0.01, 0.1, 8, 24)
    fill.Add(edge, support_face, GeomAbs_G1)
    for other in items(ruled_face, TopAbs_EDGE, TopoDS.Edge):
        if not other.IsSame(edge):
            fill.Add(other, GeomAbs_C0)
    fill.Build()
    assert fill.IsDone()
    return fill.Shape()
