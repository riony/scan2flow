"""Array/evidence-only checks from patch proof and full H0 qualification."""

import numpy as np
import trimesh

from .metrics import boundary


def topology(vertices, f):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ids = np.unique(f)
    mm = trimesh.Trimesh(vertices[ids], np.searchsorted(ids, f), process=False)
    bb = boundary(f)
    degree = {}
    graph = {}
    for a, b in bb:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    seen = set()
    loops = 0
    for a in graph:
        if a in seen:
            continue
        loops += 1
        todo = [a]
        while todo:
            b = todo.pop()
            if b in seen:
                continue
            seen.add(b)
            todo.extend(graph[b] - seen)
    return {
        "vertices": len(ids),
        "faces": len(f),
        "euler": int(mm.euler_number),
        "boundary_edges": len(bb),
        "boundary_components": loops,
        "boundary_degree_two": all(n == 2 for n in degree.values()),
        "components": int(
            connected_components(
                coo_matrix(
                    (
                        np.ones(len(mm.face_adjacency) * 2),
                        (
                            np.r_[mm.face_adjacency[:, 0], mm.face_adjacency[:, 1]],
                            np.r_[mm.face_adjacency[:, 1], mm.face_adjacency[:, 0]],
                        ),
                    ),
                    shape=(len(f), len(f)),
                ).tocsr(),
                directed=False,
                return_labels=False,
            )
        ),
        "winding_consistent": bool(mm.is_winding_consistent),
    }


def audit_seam(faces, shared):
    """Require two opposite incidences for each declared shared atomic edge."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    seam = []
    edge_map = {}
    for x, y in directed:
        edge_map.setdefault(tuple(sorted((int(x), int(y)))), []).append((int(x), int(y)))
    for a, b in shared:
        incidences = edge_map.get((int(a), int(b)), [])
        seam.append(
            {
                "edge": [int(a), int(b)],
                "incidences": incidences,
                "two_opposite": len(incidences) == 2 and incidences[0] == incidences[1][::-1],
            }
        )
    return {
        "edges": seam,
        "all_two_opposite": all(x["two_opposite"] for x in seam),
        "shared_global_vertices": len(np.unique(shared)),
    }


def reconcile_surface_checks(sf, exact, raw_reported, hit):
    """Check saved full-body evidence, with no tool invocation or blanket contact waiver."""
    structural = (
        sf["process_success"]
        and sf["complete"]
        and not sf["fatal_error"]
        and sf["illegal_triangles"] == 0
        and sf["closed"]
        and sf["components"] == 1
        and sf["normal_zones"] == 1
        and sf["min_triangle_quality"] is not None
        and sf["min_triangle_quality"] > 0.05
        and sf["intersection_check_present"]
    )
    recon = (
        exact["canonical"]["intersecting_pairs"] == 0
        and exact["reader"]["intersecting_pairs"] == 0
        and raw_reported is not None
        and hit.get("reported") == raw_reported
        and hit.get("exact_segment_triangle_intersections") == 0
        and hit.get("exact_disjoint") == hit.get("reported")
        and hit.get("degenerate_pairs") == 0
    )
    return {
        "OpenFOAM_structure_quality": bool(structural),
        "exact_intersection_reconciliation": bool(recon),
    }


def audit_locked_graph(faces, face_patch, locked, source_faces):
    """Full-H0 locked graph check, requiring incidence from distinct patches."""
    locked_set = {tuple(sorted(e)) for e in locked}
    incidences = {e: [] for e in locked_set}
    all_edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    for fi, face in enumerate(faces):
        for x, y in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            e = tuple(sorted((int(x), int(y))))
            if e in incidences:
                incidences[e].append([int(x), int(y), int(face_patch[fi])])
    bad = []
    for e, h in incidences.items():
        if not (len(h) == 2 and h[0][:2] == h[1][:2][::-1] and h[0][2] != h[1][2]):
            bad.append({"edge": e, "incidences": h})
    return {
        "locked_segments": len(locked_set),
        "shared_vertices": len(np.unique(locked)),
        "bad_seams": bad,
        "all_two_opposite_distinct_patch": not bad,
        "all_mesh_edges_two_faces": bool(np.all(counts == 2)),
        "all_global_ids_retained": set(faces.ravel()) == set(source_faces.ravel()),
    }
