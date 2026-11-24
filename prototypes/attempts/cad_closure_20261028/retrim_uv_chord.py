"""Rejected UV endpoint-chord retrim, retained for diagnostic reproduction."""

from .cad_common import items


def retrim(shape, edge_face_pairs):
    """Replace caller-selected edge trims on their carriers; may produce invalid B-rep."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Sewing
    from OCP.BRepLib import BRepLib
    from OCP.collections import Array1_double, Array1_gp_Pnt2d, Array1_int
    from OCP.Geom2d import Geom2d_BSplineCurve
    from OCP.ShapeBuild import ShapeBuild_ReShape
    from OCP.TopAbs import TopAbs_FACE

    replacement = ShapeBuild_ReShape()
    for edge, face in edge_face_pairs:
        first, last = BRep_Tool.Range_s(edge)
        curve = BRep_Tool.CurveOnSurface_s(edge, face, first, last)
        poles = Array1_gp_Pnt2d(1, 2)
        poles.SetValue(1, curve.Value(first))
        poles.SetValue(2, curve.Value(last))
        knots = Array1_double(1, 2)
        knots.SetValue(1, 0)
        knots.SetValue(2, 1)
        multiplicities = Array1_int(1, 2)
        multiplicities.SetValue(1, 2)
        multiplicities.SetValue(2, 2)
        segment = Geom2d_BSplineCurve(poles, knots, multiplicities, 1)
        new = BRepBuilderAPI_MakeEdge(segment, BRep_Tool.Surface_s(face)).Edge()
        BRepLib.BuildCurves3d_s(new)
        new.Orientation(edge.Orientation())
        replacement.Replace(edge, new)
    sewing = BRepBuilderAPI_Sewing(1e-5)
    for face in items(replacement.Apply(shape), TopAbs_FACE):
        sewing.Add(face)
    # The original retrim sewed surfaces only; it did not make a solid.
    sewing.Perform()
    return sewing.SewedShape()
