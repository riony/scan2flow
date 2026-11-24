"""Bounded planar-patch retriangulation, immutable boundary graph and coordinates."""

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, Delaunay


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


def patch(vertices, original_faces, max_rounds=12):
    ids = np.unique(original_faces)
    original_boundary = boundary(original_faces)
    fixed = set(x for edge in original_boundary for x in edge)
    xyz = vertices[ids]
    centre = xyz.mean(0)
    _, _, vh = np.linalg.svd(xyz - centre, full_matrices=False)
    error = np.abs((xyz - centre) @ vh[2]).max()
    if error > 1e-7:
        raise ValueError(f"Nonplanar patch: {error}")
    xy = (xyz - centre) @ vh[:2].T
    area = trimesh.Trimesh(vertices, original_faces, process=False).area
    if abs(ConvexHull(xy).volume - area) > max(1e-10, area * 1e-6):
        raise ValueError("Nonconvex patch: unconstrained Delaunay not applicable")
    normal = np.cross(
        vertices[original_faces[:, 1]] - vertices[original_faces[:, 0]],
        vertices[original_faces[:, 2]] - vertices[original_faces[:, 0]],
    ).sum(0)
    keep = np.ones(len(ids), dtype=bool)
    rounds = []
    for iteration in range(max_rounds + 1):
        active = ids[keep]
        tri = active[Delaunay(xy[keep]).simplices]
        cross = np.cross(
            vertices[tri[:, 1]] - vertices[tri[:, 0]], vertices[tri[:, 2]] - vertices[tri[:, 0]]
        )
        flip = cross @ normal < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        if boundary(tri) != original_boundary:
            raise ValueError("Delaunay did not preserve every constrained boundary edge")
        q = quality(vertices, tri)
        bad = tri[q <= 0.05]
        rounds.append({"round": iteration, "bad_faces": len(bad), "min_quality": float(q.min())})
        if not len(bad) or iteration == max_rounds:
            break
        remove = set()
        for face in bad:
            options = []
            for v in face:
                if int(v) not in fixed:
                    dist = np.linalg.norm(vertices[face[face != v]] - vertices[v], axis=1)
                    options.append((float(dist.min()), int(v)))
            if options:
                remove.add(min(options)[1])
        if not remove:
            break
        keep &= ~np.isin(ids, list(remove))
    return tri, {
        "plane_max_error_m": float(error),
        "boundary_edges": len(fixed),
        "removed_interior_vertices": int((~keep).sum()),
        "rounds": rounds,
    }


def repair(mesh, locked_edges):
    locked = {tuple(sorted(e)) for e in locked_edges}
    adjacency = mesh.face_adjacency
    take = np.array([tuple(sorted(e)) not in locked for e in mesh.face_adjacency_edges])
    a = adjacency[take]
    graph = coo_matrix(
        (np.ones(len(a) * 2), (np.r_[a[:, 0], a[:, 1]], np.r_[a[:, 1], a[:, 0]])),
        shape=(len(mesh.faces), len(mesh.faces)),
    ).tocsr()
    n, labels = connected_components(graph, directed=False)
    results, records = [], []
    for k in range(n):
        f, record = patch(mesh.vertices, mesh.faces[labels == k])
        results.append(f)
        records.append(record)
    faces = np.vstack(results)
    edges = {
        tuple(e)
        for e in np.sort(trimesh.Trimesh(mesh.vertices, faces, process=False).edges_unique, axis=1)
    }
    if not locked.issubset(edges):
        raise ValueError("Locked graph lost")
    result = trimesh.Trimesh(mesh.vertices.copy(), faces, process=False)
    return result, {
        "patches": records,
        "patch_count": n,
        "locked_edges": len(locked),
        "locked_graph_preserved": True,
        "vertex_positions_unchanged": bool(np.array_equal(result.vertices, mesh.vertices)),
        "quality_passed": bool(np.all(quality(result.vertices, result.faces) > 0.05)),
    }
