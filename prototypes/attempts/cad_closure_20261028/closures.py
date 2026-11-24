"""Historical diagnostic closures. B-rep validity does not establish nonintersection."""

import numpy as np

from .cad_common import finish, free_edges, items, mirror, sample


def reflected_bridge(shape, skip_centreline=False):
    """Initial ruled reflection bridge; known historical self-intersection failure."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.BRepFill import BRepFill
    from OCP.TopoDS import TopoDS

    sewing = BRepBuilderAPI_Sewing(1e-5)
    sewing.Add(shape)
    sewing.Add(mirror(shape))
    for edge in free_edges(shape):
        if skip_centreline and np.max(np.abs(sample(edge)[:, 1])) < 1e-6:
            continue
        sewing.Add(BRepFill.Face_s(edge, TopoDS.Edge(mirror(edge))))
    return finish(sewing) if skip_centreline else _finish_initial_bridge(sewing)


def _finish_initial_bridge(sewing):
    """Initial attempt asserted one shell, then made a solid unconditionally."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepLib import BRepLib
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS

    sewing.Perform()
    shells = items(sewing.SewedShape(), TopAbs_SHELL, TopoDS.Shell)
    assert len(shells) == 1
    solid = BRepBuilderAPI_MakeSolid(shells[0]).Solid()
    BRepLib.OrientClosedSolid_s(solid)
    return solid, {"free_edges": sewing.NbFreeEdges(), "multiple_edges": sewing.NbMultipleEdges()}


def clip_half(shape, lower, upper):
    """Clip against caller-specified bounds; no physical-datum validation."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    box = BRepPrimAPI_MakeBox(gp_Pnt(*lower), gp_Pnt(*upper)).Shape()
    operation = BRepAlgoAPI_Common(shape, box)
    operation.Build()
    assert operation.IsDone()
    return operation.Shape()


def _named_join(shape, orient_partner):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Sewing
    from OCP.BRepFill import BRepFill
    from OCP.gp import gp_Pnt

    sewing = BRepBuilderAPI_Sewing(1e-5)
    sewing.Add(shape)
    sewing.Add(mirror(shape))
    for edge in free_edges(shape):
        points = sample(edge)
        if np.max(np.abs(points[:, 1])) < 1e-6:
            continue
        target_points = points[[0, -1]].copy()
        target_points[:, 1] = 0
        target = BRepBuilderAPI_MakeEdge(
            gp_Pnt(*target_points[0]), gp_Pnt(*target_points[1])
        ).Edge()
        if orient_partner:
            target.Orientation(edge.Orientation())
        face = BRepFill.Face_s(edge, target)
        sewing.Add(face)
        sewing.Add(mirror(face))
    return finish(sewing)


def named_join_legacy(shape):
    """Rejected initial implementation: partner orientation is not matched."""
    return _named_join(shape, False)


def named_join_oriented(shape):
    """Connectivity correction; historical candidate still failed intersections."""
    return _named_join(shape, True)


def partition(shape):
    """Historical volume partition recipe; caller must bound runtime externally."""
    from OCP.BOPAlgo import BOPAlgo_MakerVolume
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS

    operation = BOPAlgo_MakerVolume()
    operation.SetIntersect(True)
    operation.SetAvoidInternalShapes(True)
    operation.SetNonDestructive(True)
    operation.SetRunParallel(False)
    operation.SetFuzzyValue(1e-7)
    for face in items(shape, TopAbs_FACE, TopoDS.Face):
        operation.AddArgument(face)
    operation.Perform()
    if operation.HasErrors():
        raise RuntimeError("Partition failed")
    return operation.Shape()
