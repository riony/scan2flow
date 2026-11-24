"""Unsigned proximity samples, not physical boundary matching or containment proof."""

import numpy as np

from ..cad_closure_20261028.cad_common import sample


class UnsignedDistance:
    def __init__(self, mesh):
        import pyvista as pv
        import vtk

        triangles = np.asarray(mesh.vertices)[mesh.faces]
        area2 = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        faces = np.asarray(mesh.faces)[area2 > 1e-14]
        assert len(faces) > 0
        self.poly = pv.PolyData(
            np.asarray(mesh.vertices), np.column_stack([np.full(len(faces), 3), faces]).ravel()
        )
        self.loc = vtk.vtkStaticCellLocator()
        self.loc.SetDataSet(self.poly)
        self.loc.BuildLocator()
        self.point = [0.0, 0.0, 0.0]
        self.cell, self.sub, self.d2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)

    def EvaluateFunction(self, point):
        self.loc.FindClosestPoint(point, self.point, self.cell, self.sub, self.d2)
        return float(self.d2.get()) ** 0.5


def values(function, points):
    return np.array([abs(function.EvaluateFunction(point)) for point in points])


def stats(values):
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def arc_sample(edge, count=201):
    points = sample(edge, 2001)
    stations = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    query = np.linspace(0, stations[-1], count)
    return np.column_stack([np.interp(query, stations, points[:, j]) for j in range(3)])
