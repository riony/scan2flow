"""Synthetic inverted-face negative control for the optional STEP verifier."""

import re

import numpy as np
import trimesh

from .compact_step import write_step
from .verify_facets import read_preserving_orientation


def check_orientation(shape, mesh):
    """Retained independent normal check, including when BRepCheck is false."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from scipy.spatial import cKDTree

    tree = cKDTree(mesh.triangles_center)
    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    mismatch = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        centre = props.CentreOfMass()
        _, index = tree.query([centre.X(), centre.Y(), centre.Z()])
        direction = BRepAdaptor_Surface(face).Plane().Axis().Direction()
        normal = np.array([direction.X(), direction.Y(), direction.Z()])
        normal *= -1 if face.Orientation() == TopAbs_REVERSED else 1
        mismatch += np.dot(normal, mesh.face_normals[index]) < 1 - 1e-10
        explorer.Next()
    return {
        "BRep_valid": valid,
        "normal_mismatches": int(mismatch),
        "passes": bool(valid and mismatch == 0),
    }


def run(output_directory):
    """Create a tiny closed fixture and refuse its intentionally inverted face."""
    from pathlib import Path

    directory = Path(output_directory)
    directory.mkdir(exist_ok=False)
    mesh = trimesh.creation.icosphere(subdivisions=3)
    source, bad = directory / "control.step", directory / "inverted.step"
    write_step(source, mesh.vertices, mesh.faces)
    changed, count = re.subn(
        r"(FACE_SURFACE\([^\n]+),\.T\.\);", r"\1,.F.);", source.read_text(), count=1
    )
    assert count == 1
    bad.write_text(changed)
    control = check_orientation(read_preserving_orientation(source, source), mesh)
    inverted = check_orientation(read_preserving_orientation(bad, source), mesh)
    assert control["passes"]
    assert not inverted["passes"] and inverted["normal_mismatches"] > 0
    return {"control": control, "inverted_face": inverted}
