"""Array/locator helpers for historical mesh controls; no archive discovery."""

import json
from pathlib import Path

import numpy as np


def save(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist(),
        )
        + "\n"
    )


def write_off(mesh, path):
    mesh.export(str(path), file_type="off")


def distance_segments(points, segments):
    points = np.asarray(points)
    best = np.full(len(points), np.inf)
    for a, b in segments:
        vector = b - a
        squared = vector @ vector
        parameter = (
            np.clip(((points - a) @ vector) / squared, 0, 1)
            if squared > 0
            else np.zeros(len(points))
        )
        delta = points - a - parameter[:, None] * vector
        best = np.minimum(best, np.einsum("ij,ij->i", delta, delta))
    return np.sqrt(best)


def surface_locator(mesh):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices), deep=True))
    cells = vtk.vtkCellArray()
    for triangle in mesh.faces:
        cells.InsertNextCell(3, triangle)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    return locator, poly


def closest(locator, points):
    import vtk

    out, distances = np.empty_like(points, dtype=float), np.empty(len(points))
    query = [0.0, 0.0, 0.0]
    cell, sub, distance = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    for index, point in enumerate(points):
        locator.FindClosestPoint(point, query, cell, sub, distance)
        out[index] = query
        distances[index] = float(distance.get()) ** 0.5
    return out, distances
