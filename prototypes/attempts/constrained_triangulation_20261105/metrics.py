"""Deterministic mesh metrics retained from the saved H0 qualification.

Sampled distances are not continuous surface-distance certificates. VTK is
needed only by the explicit surface-locator functions, never at import time.
"""

import numpy as np


def quality(vertices, faces):
    t = vertices[faces]
    a = t[:, 1] - t[:, 0]
    b = t[:, 2] - t[:, 0]
    c = t[:, 2] - t[:, 1]
    area = np.linalg.norm(np.cross(a, b), axis=1) / 2
    den = (a * a).sum(1) * (b * b).sum(1) * (c * c).sum(1)
    return np.divide(64 * area**3, 3 * np.sqrt(3) * den, out=np.zeros(len(faces)), where=den > 0)


def boundary(faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, count = np.unique(edges, axis=0, return_counts=True)
    if np.any(count > 2):
        raise ValueError("Nonmanifold patch")
    return {tuple(e) for e in edges[count == 1]}


def distance_segments(points, segments):
    # Bounded-memory exact distance to all original feature segments.
    points = np.asarray(points)
    best = np.full(len(points), np.inf)
    for a, b in segments:
        v = b - a
        vv = v @ v
        t = np.clip(((points - a) @ v) / vv, 0, 1) if vv > 0 else np.zeros(len(points))
        d = points - a - t[:, None] * v
        best = np.minimum(best, np.einsum("ij,ij->i", d, d))
    return np.sqrt(best)


def sampled_surface(mesh, count=200000, seed=311026):
    rng = np.random.default_rng(seed)
    tri = mesh.triangles
    area = mesh.area_faces
    cum = np.cumsum(area)
    idx = np.searchsorted(cum, (np.arange(count) + rng.random(count)) * cum[-1] / count)
    uv = rng.random((count, 2))
    flip = uv.sum(1) > 1
    uv[flip] = 1 - uv[flip]
    return (
        tri[idx, 0]
        + uv[:, 0, None] * (tri[idx, 1] - tri[idx, 0])
        + uv[:, 1, None] * (tri[idx, 2] - tri[idx, 0])
    )


def surface_locator(mesh):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    p = vtk.vtkPoints()
    p.SetData(numpy_to_vtk(np.asarray(mesh.vertices), deep=True))
    cells = vtk.vtkCellArray()
    for t in mesh.faces:
        cells.InsertNextCell(3, t)
    poly = vtk.vtkPolyData()
    poly.SetPoints(p)
    poly.SetPolys(cells)
    loc = vtk.vtkStaticCellLocator()
    loc.SetDataSet(poly)
    loc.BuildLocator()
    return loc, poly


def closest(loc, points):
    import vtk

    out = np.empty_like(points, dtype=float)
    ds = np.empty(len(points))
    q = [0.0, 0.0, 0.0]
    cell = vtk.reference(0)
    sub = vtk.reference(0)
    dist = vtk.reference(0.0)
    for i, p in enumerate(points):
        loc.FindClosestPoint(p, q, cell, sub, dist)
        out[i] = q
        ds[i] = float(dist.get()) ** 0.5
    return out, ds


def surface_samples(m):
    return np.vstack(
        [m.vertices, m.triangles.mean(1), m.vertices[m.edges_unique].mean(1), sampled_surface(m)]
    )


def stats(x):
    return {
        "max": float(np.max(x)),
        "p90": float(np.quantile(x, 0.9)),
        "median": float(np.median(x)),
    }


def feature_metrics(mesh, points, ids):
    loc, _ = surface_locator(mesh)
    q, d = closest(loc, points)
    angles = []
    for k in np.unique(ids):
        p = points[ids == k]
        qq = q[ids == k]
        aa = np.diff(p, axis=0)
        bb = np.diff(qq, axis=0)
        den = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
        valid = den > 1e-20
        cos = np.full(len(aa), -1.0)
        cos[valid] = np.clip(np.einsum("ij,ij->i", aa[valid], bb[valid]) / den[valid], -1, 1)
        angles.extend(np.degrees(np.arccos(cos)))
    return {"distance_m": stats(d), "mapped_tangent_change_deg": stats(np.asarray(angles))}, loc


def component_count(m):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a = m.face_adjacency
    g = coo_matrix(
        (np.ones(2 * len(a)), (np.r_[a[:, 0], a[:, 1]], np.r_[a[:, 1], a[:, 0]])),
        shape=(len(m.faces), len(m.faces)),
    ).tocsr()
    return int(connected_components(g, directed=False, return_labels=False))
