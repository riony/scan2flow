"""Validate caller-prescribed convex chart domains; no inferred closure recipe."""

import numpy as np


def inspect_chart(mesh, *, name, donors, loop, axes, normal, planar):
    """Check a supplied chart; the caller owns its prescription and source provenance."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    m = mesh
    v = m.vertices
    f = m.faces
    ids = np.unique(np.r_[f[donors].reshape(-1), loop]).astype(int)
    points = v[ids][:, axes]
    poly = Polygon(v[loop][:, axes])
    assert poly.is_valid and poly.area > 0
    assert abs(poly.area - poly.convex_hull.area) < 1e-12
    assert len(np.unique(points, axis=0)) == len(ids)
    orig = [Polygon(v[t][:, axes]) for t in f[donors]]
    union = unary_union(orig)
    area = sum(x.area for x in orig)
    assert abs(area - union.area) < 1e-12 and union.difference(poly).area < 1e-12
    p = v[ids]
    _, _, vh = np.linalg.svd(p - p.mean(0), full_matrices=False)
    err = float(np.abs((p - p.mean(0)) @ vh[-1]).max())
    if planar:
        assert err < 1e-7
    else:
        assert np.all(m.face_normals[donors] @ normal > 0)
    edges = {tuple(sorted((a, b))) for face in f[donors] for a, b in zip(face, np.roll(face, -1))}
    edges.update(tuple(sorted((a, b))) for a, b in zip(loop, loop[1:] + loop[:1]))
    edges = sorted(edges)
    missing = poly.difference(union)
    domains = (
        list(missing.geoms)
        if hasattr(missing, "geoms")
        else ([] if missing.is_empty else [missing])
    )
    record = {
        "name": name,
        "donor_face_ids": donors,
        "outer_boundary_ids": loop,
        "projection_axes": axes,
        "outward_direction": normal,
        "planar": planar,
        "plane_error_m": err,
        "vertices": len(ids),
        "constraints": len(edges),
        "domain_area_m2": poly.area,
        "donor_projected_area_m2": union.area,
        "missing_projected_area_m2": missing.area,
        "missing_domains": len(domains),
        "no_donor_overlap": True,
        "donor_within_domain": True,
        "injective_donor_projection": True,
        "new_faces_provenance": "prescribed_" + name,
    }
    return {"record": record, "vertex_ids": ids, "points_2d": points, "constraints": edges}


def cap_ownership_conflicts(donor_faces, cap_faces):
    """Return donor IDs that a proposed cap duplicates, regardless of orientation."""
    owned = {tuple(sorted(face)): i for i, face in enumerate(donor_faces)}
    return [owned[tuple(sorted(face))] for face in cap_faces if tuple(sorted(face)) in owned]
