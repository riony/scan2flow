"""Retained prior-to-donor transfer kernel, without geometry or tool orchestration.

The caller provides a closed carrier, declared feature segments, both donor
projection targets and surviving-donor distances. The distance-based composite
has no normal-qualified support test. Its finite guard is not qualification.
"""

import numpy as np
import trimesh

from ..constrained_triangulation_20261105.metrics import distance_segments
from ..constrained_triangulation_20261105.metrics import quality as triangle_qualities


def transfer(carrier, segments, mapped_vdb, mapped_surv, dist_surv):
    v = carrier.vertices.copy()
    base = v.copy()
    d_feat = distance_segments(v, segments)
    t = np.clip((d_feat - 0.013) / 0.013, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)

    # 5. Query both OpenVDB donor (for missing closure regions & global level-set support) and surviving donor (for observed panels)

    # Smoothly transition from surviving donor on observed support (dist_surv <= 1.3 mm) to OpenVDB guide in closure holes (dist_surv >= 3.9 mm)
    s_hole = np.clip((dist_surv - 0.0013) / 0.0026, 0.0, 1.0)
    w_hole = s_hole * s_hole * (3.0 - 2.0 * s_hole)
    mapped = (1.0 - w_hole)[:, None] * mapped_surv + w_hole[:, None] * mapped_vdb
    mapped[w == 0] = base[w == 0]

    # 6. Verify/guard unconstrained crease angles (< 27.5 deg) and triangle qualities (> 0.075 in float64 and float32) on hybrid
    adj_edges = carrier.face_adjacency_edges
    edge_mids = base[adj_edges].mean(1)
    unconstrained_edge_mask = distance_segments(edge_mids, segments) > 0.0004
    unconstrained_edges = adj_edges[unconstrained_edge_mask]

    # Build vertex-to-vertex adjacency for 1-ring damping when needed
    v_neighbors = [[] for _ in range(len(v))]
    for a_idx, b_idx in adj_edges:
        v_neighbors[a_idx].append(b_idx)
        v_neighbors[b_idx].append(a_idx)

    guard_iters = 0
    for _ in range(30):
        result = v + w[:, None] * (mapped - v)
        result[w == 0] = base[w == 0]
        m_try = trimesh.Trimesh(result, carrier.faces, process=False)
        ang_deg = np.degrees(m_try.face_adjacency_angles)
        bad_unconstrained = unconstrained_edge_mask & (ang_deg >= 27.4)
        q64 = triangle_qualities(result, carrier.faces)
        q32 = triangle_qualities(result.astype(np.float32).astype(np.float64), carrier.faces)
        bad_faces = (q64 <= 0.075) | (q32 <= 0.075)
        if not np.any(bad_unconstrained) and not np.any(bad_faces):
            break
        guard_iters += 1
        bad_verts = set()
        if np.any(bad_unconstrained):
            for u_idx in adj_edges[bad_unconstrained].ravel():
                bad_verts.add(int(u_idx))
                for nb in v_neighbors[int(u_idx)]:
                    bad_verts.add(int(nb))
        if np.any(bad_faces):
            for u_idx in carrier.faces[bad_faces].ravel():
                bad_verts.add(int(u_idx))
                for nb in v_neighbors[int(u_idx)]:
                    bad_verts.add(int(nb))
        for u_idx in bad_verts:
            if w[u_idx] > 0:
                mapped[u_idx] = v[u_idx] + 0.5 * (mapped[u_idx] - v[u_idx])

    result = v + w[:, None] * (mapped - v)
    result[w == 0] = base[w == 0]
    hybrid = trimesh.Trimesh(result, carrier.faces.copy(), process=False)

    fw = w[carrier.faces]
    transfer_tags = np.full(len(carrier.faces), 2, dtype=np.uint8)
    transfer_tags[np.all(fw == 0, axis=1)] = 0
    transfer_tags[np.all(fw == 1, axis=1)] = 1
    transfer_names = ["retained_carrier", "donor_guided", "join_blended"]
    return {
        "hybrid": hybrid,
        "vertex_weight": w,
        "vertex_hole_weight": w_hole,
        "donor_target": mapped,
        "face_transfer_class": transfer_tags,
        "transfer_class_names": transfer_names,
        "guard_iterations": guard_iters,
    }
