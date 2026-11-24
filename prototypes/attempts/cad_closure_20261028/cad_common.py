"""Small CAD operations from the retained closure attempts; OCP is optional."""

import numpy as np


def items(shape, kind, cast=lambda value: value):
    from OCP.TopExp import TopExp_Explorer

    explorer = TopExp_Explorer(shape, kind)
    result = []
    while explorer.More():
        result.append(cast(explorer.Current()))
        explorer.Next()
    return result


def free_edges(shape):
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as Map,
    )
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS

    mapping = Map()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, mapping)
    return [
        TopoDS.Edge(mapping.FindKey(i))
        for i in range(1, mapping.Extent() + 1)
        if mapping.FindFromIndex(i).Size() == 1
    ]


def sample(edge, count=101):
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    curve = BRepAdaptor_Curve(edge)
    points = [
        curve.Value(float(t))
        for t in np.linspace(curve.FirstParameter(), curve.LastParameter(), count)
    ]
    return np.array([[p.X(), p.Y(), p.Z()] for p in points])


def mirror(shape):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf

    transform = gp_Trsf()
    transform.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def finish(sewing):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepLib import BRepLib
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS

    sewing.Perform()
    shape = sewing.SewedShape()
    shells = items(shape, TopAbs_SHELL, TopoDS.Shell)
    if len(shells) == 1 and sewing.NbFreeEdges() == 0:
        shape = BRepBuilderAPI_MakeSolid(shells[0]).Solid()
        BRepLib.OrientClosedSolid_s(shape)
    return shape, {"free_edges": sewing.NbFreeEdges(), "multiple_edges": sewing.NbMultipleEdges()}
