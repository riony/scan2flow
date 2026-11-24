"""Closed-carrier donor transfer from the failed constrained-hybrid coupon."""

import numpy as np
import trimesh

from ..wrap_controls_20261031.control_common import closest, distance_segments, surface_locator


def transfer(carrier, donor, segments, stressed_vertices=None):
    """Transfer with the original .013/.026 fixed-strip smoothstep and connectivity.

    Input units must match those distances. A closed carrier is a method
    prerequisite; this function does not construct one from disconnected panels.
    Optional stressed vertices replace the archived synthetic stress recipe.
    """
    base = carrier.vertices.copy()
    vertices = base.copy() if stressed_vertices is None else np.array(stressed_vertices).copy()
    distance = distance_segments(base, segments)
    parameter = np.clip((distance - 0.013) / 0.013, 0, 1)
    weight = parameter * parameter * (3 - 2 * parameter)
    locator, _ = surface_locator(donor)
    mapped, before = closest(locator, vertices)
    result = vertices + weight[:, None] * (mapped - vertices)
    result[weight == 0] = base[weight == 0]
    hybrid = trimesh.Trimesh(result, carrier.faces.copy(), process=False)
    face_weight = weight[carrier.faces]
    tags = np.full(len(carrier.faces), 2, dtype=np.uint8)
    tags[np.all(face_weight == 0, axis=1)] = 0
    tags[np.all(face_weight == 1, axis=1)] = 1
    return hybrid, {
        "vertex_weight": weight,
        "face_class": tags,
        "carrier_vertex": base,
        "donor_target": mapped,
        "donor_distance_before": before,
        "hybrid_vertex": result,
        "faces": carrier.faces.copy(),
    }
