"""Positive-winding patch decisions from the exact-split exterior diagnostic."""

import numpy as np


def winding(triangles, point):
    total = 0.0
    for block in np.array_split(triangles, 8):
        vectors = block - point
        norm = np.linalg.norm(vectors, axis=2)
        vectors = vectors / np.maximum(norm[:, :, None], 1e-300)
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        numerator = np.einsum("ij,ij->i", x, np.cross(y, z))
        denominator = (
            1
            + np.einsum("ij,ij->i", x, y)
            + np.einsum("ij,ij->i", y, z)
            + np.einsum("ij,ij->i", z, x)
        )
        total += np.arctan2(numerator, denominator).sum() * 2
    return float(total / (4 * np.pi))


def classify_patch(reference_triangles, triangles, areas, centres, normals, face_ids):
    """Return +1/-1/0, or None when two largest usable samples disagree."""
    chosen = face_ids[np.argsort(areas[face_ids])[-2:]]
    votes, samples = [], []
    for face in chosen:
        triangle = triangles[face]
        altitude = (
            2
            * areas[face]
            / max(np.linalg.norm(np.roll(triangle, 1, axis=0) - triangle, axis=1).max(), 1e-30)
        )
        epsilon = min(1e-6, altitude * 0.01)
        if epsilon < 1e-14:
            continue
        minus = winding(reference_triangles, centres[face] - epsilon * normals[face])
        plus = winding(reference_triangles, centres[face] + epsilon * normals[face])
        inside_minus, inside_plus = minus > 0.5, plus > 0.5
        votes.append(
            1
            if inside_minus and not inside_plus
            else (-1 if inside_plus and not inside_minus else 0)
        )
        samples.append({"face": int(face), "epsilon": float(epsilon), "minus": minus, "plus": plus})
    decision = votes[0] if votes and len(set(votes)) == 1 else None
    return decision, samples
