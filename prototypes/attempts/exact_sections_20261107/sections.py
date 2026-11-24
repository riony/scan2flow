"""Historical float64 triangle-plane sections, connected by source topology.

Every intersecting branch is retained. This is not exact-predicate arithmetic,
physical accuracy certification or a cleanup/branch-selection algorithm.
"""

from collections import defaultdict

import numpy as np
import trimesh


def cut_mesh(vertices, faces, origin, normal, eps=1e-12):
    """Retain triangle owners, including duplicate ownership on on-plane edges.

    Nodes are source vertices (0,v,v), or source edges (1,min(v),max(v)).
    Completely coplanar triangles are retained separately, not reduced to curves.
    """
    normal = normal / np.linalg.norm(normal)
    distances = (vertices - origin) @ normal
    nodes, node_xyz = {}, []
    edges, owners, edge_lookup = [], [], {}
    point_contacts, coplanar = [], []
    zero_length_faces = []
    generic_faces = []
    generic_segments = []
    coordinate_repeat_max = 0.0

    def node(key, xyz):
        nonlocal coordinate_repeat_max
        key = tuple(key)
        if key not in nodes:
            nodes[key] = len(node_xyz)
            node_xyz.append(xyz)
        else:
            coordinate_repeat_max = max(
                coordinate_repeat_max, float(np.linalg.norm(node_xyz[nodes[key]] - xyz))
            )
        return nodes[key]

    for start in range(0, len(faces), 250000):
        chunk = faces[start : start + 250000]
        dd = distances[chunk]
        chosen = np.flatnonzero((dd.min(1) <= eps) & (dd.max(1) >= -eps))
        for j in chosen:
            fid = start + int(j)
            ids, ds = chunk[j], dd[j]
            on = np.abs(ds) <= eps
            if on.all():
                coplanar.append(fid)
                continue
            hits = {}
            for v in ids[on]:
                hits[(0, int(v), int(v))] = vertices[v].copy()
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if on[a] or on[b] or ds[a] * ds[b] >= 0:
                    continue
                va, vb = sorted((int(ids[a]), int(ids[b])))
                da, db = distances[va], distances[vb]
                hits[(1, va, vb)] = vertices[va] + (da / (da - db)) * (vertices[vb] - vertices[va])
            if len(hits) == 1:
                k, x = next(iter(hits.items()))
                point_contacts.append({"face_id": fid, "key": list(k), "xyz": x.tolist()})
                continue
            if len(hits) != 2:
                raise RuntimeError(f"Unresolved triangle-plane case: face={fid}, hits={len(hits)}")
            nk = [node(k, x) for k, x in hits.items()]
            seg = np.array([node_xyz[k] for k in nk])
            if np.linalg.norm(seg[1] - seg[0]) <= eps:
                zero_length_faces.append(fid)
            key = tuple(sorted(nk))
            if key in edge_lookup:
                owners[edge_lookup[key]].append(fid)
            else:
                edge_lookup[key] = len(edges)
                edges.append(nk)
                owners.append([fid])
            if not on.any():
                generic_faces.append(fid)
                generic_segments.append(seg)

    xyz = np.asarray(node_xyz, dtype=float).reshape(-1, 3)
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    adjacency = defaultdict(list)
    for i, (a, b) in enumerate(edges):
        adjacency[a].append(i)
        adjacency[b].append(i)
    used = np.zeros(len(edges), bool)
    branches = []

    def walk(first_node, first_edge):
        ids, verts = [], [int(first_node)]
        v, e = first_node, first_edge
        while not used[e]:
            used[e] = True
            ids.append(int(e))
            a, b = edges[e]
            v = b if v == a else a
            verts.append(int(v))
            if len(adjacency[v]) != 2:
                break
            options = [x for x in adjacency[v] if not used[x]]
            if not options:
                break
            e = options[0]
        branches.append({"edge_ids": ids, "node_ids": verts, "closed": bool(verts[0] == verts[-1])})

    for v in sorted(adjacency):
        if len(adjacency[v]) != 2:
            for e in adjacency[v]:
                if not used[e]:
                    walk(v, e)
    for e, (a, _) in enumerate(edges):
        if not used[e]:
            walk(a, e)
    assert used.all()
    assert sorted(e for b in branches for e in b["edge_ids"]) == list(range(len(edges)))
    # Independent implementation, same literal planes, all transverse faces.
    independent_error = 0.0
    if generic_faces:
        sub = trimesh.Trimesh(vertices=vertices, faces=faces[generic_faces], process=False)
        ref, fid = trimesh.intersections.mesh_plane(sub, normal, origin, return_faces=True)
        assert len(ref) == len(generic_faces), (len(ref), len(generic_faces))
        assert np.array_equal(np.sort(fid), np.arange(len(generic_faces)))
        ours = np.asarray(generic_segments)[fid]
        direct = np.linalg.norm(ref - ours, axis=2).max(1)
        reverse = np.linalg.norm(ref[:, ::-1] - ours, axis=2).max(1)
        independent_error = float(np.minimum(direct, reverse).max())
    residual = float(np.abs((xyz - origin) @ normal).max()) if len(xyz) else 0.0
    deg = np.array([len(adjacency[k]) for k in range(len(xyz))])
    meta = {
        "source_triangle_intersections": sum(map(len, owners)),
        "unique_segments": len(edges),
        "nodes": len(xyz),
        "branches": len(branches),
        "closed_branches": sum(b["closed"] for b in branches),
        "open_branches": sum(not b["closed"] for b in branches),
        "junction_nodes": int((deg > 2).sum()),
        "end_nodes": int((deg == 1).sum()),
        "multiple_owner_segments": sum(len(o) > 1 for o in owners),
        "point_tangent_contacts": len(point_contacts),
        "coplanar_triangles": len(coplanar),
        "zero_length_face_intersections": len(zero_length_faces),
        "maximum_plane_residual": residual,
        "maximum_repeated_node_disagreement": coordinate_repeat_max,
        "independently_checked_transverse_faces": len(generic_faces),
        "maximum_independent_endpoint_difference": independent_error,
        "all_segments_assigned_once": True,
    }
    return {
        "xyz": xyz,
        "edges": edge_array,
        "node_keys": np.asarray(list(nodes), dtype=np.int64).reshape(-1, 3),
        "owners": owners,
        "branches": branches,
        "metadata": meta,
        "point_contacts": point_contacts,
        "coplanar_faces": coplanar,
        "zero_length_faces": zero_length_faces,
    }


def fixtures():
    box = trimesh.creation.box(extents=[2, 2, 2])
    r = cut_mesh(box.vertices, box.faces, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert r["metadata"]["closed_branches"] == 1
    s = r["xyz"][r["edges"]]
    assert abs(np.linalg.norm(s[:, 1] - s[:, 0], axis=1).sum() - 8) < 1e-12
    # Same geometric box twice, separate topology: must remain two loops.
    v = np.vstack([box.vertices, box.vertices])
    f = np.vstack([box.faces, box.faces + len(box.vertices)])
    duplicate = cut_mesh(v, f, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert duplicate["metadata"]["closed_branches"] == 2
    # One on-plane edge, a vertex tangent, a transverse on-vertex cut, coplanar facet.
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [0, -1, -1], [1, 1, 1], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 4], [0, 2, 3], [0, 1, 5]])
    special = cut_mesh(v, f, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert special["metadata"]["multiple_owner_segments"] == 1
    assert special["metadata"]["point_tangent_contacts"] == 1
    assert special["metadata"]["coplanar_triangles"] == 1
    # Three triangles share a cut edge point: junction must split branches.
    v = np.array([[0, 0, -1], [0, 0, 1], [1, 0, -1], [0, 1, -1], [-1, 0, -1]], float)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    junction = cut_mesh(v, f, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert junction["metadata"]["junction_nodes"] == 1
    assert junction["metadata"]["branches"] == 3
    return {
        "status": "passed",
        "box_perimeter": 8.0,
        "coincident_disconnected_boxes": duplicate["metadata"],
        "on_plane_degeneracies": special["metadata"],
        "junction": junction["metadata"],
    }
