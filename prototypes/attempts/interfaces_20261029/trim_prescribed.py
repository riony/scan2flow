"""Trim caller-provided prescribed solids with source-face outward prisms."""

import numpy as np

from ..cad_closure_20261028.cad_common import items, mirror


def outward_cutters(faces, triangle_vertices, triangle_face_ids, extrusion_length=10):
    """Face directions use signed area-weighted triangle normals, unchanged recipe."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS_Compound

    cross = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    builder, cutters = BRep_Builder(), TopoDS_Compound()
    builder.MakeCompound(cutters)
    for index, face in enumerate(faces):
        normal = cross[triangle_face_ids == index].sum(0)
        normal /= np.linalg.norm(normal)
        prism = BRepPrimAPI_MakePrism(
            face, gp_Vec(*(normal * extrusion_length)), False, True
        ).Shape()
        assert BRepCheck_Analyzer(prism).IsValid() and len(items(prism, TopAbs_SOLID)) == 1
        builder.Add(cutters, prism)
        builder.Add(cutters, mirror(prism))
    return cutters


def trim(draft, cutters):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

    operation = BRepAlgoAPI_Cut(draft, cutters)
    operation.SetRunParallel(False)
    operation.SetNonDestructive(True)
    operation.Build()
    assert operation.IsDone()
    return operation.Shape()
