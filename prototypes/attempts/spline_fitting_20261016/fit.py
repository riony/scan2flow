"""Historical spatial-median, robust uniform bicubic height-field fit.

Callers supply already selected chart points. Domain segmentation, chart
construction, physical boundaries, units and acceptance are outside this API.
The sampled bending objective is preserved, not numerically modernized.
"""

import numpy as np
from scipy.interpolate import BSpline


def spatial_medians(points, lo, cell=0.055, min_points=3):
    """Give each occupied XY cell one median, including its median height."""
    ids = np.floor((points[:, :2] - lo) / cell).astype(int)
    unique, inverse, counts = np.unique(ids, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse)
    groups = np.split(order, np.cumsum(counts)[:-1])
    centers = np.array([np.median(points[g], axis=0) for g in groups if len(g) >= min_points])
    bins = np.array([unique[i] for i, g in enumerate(groups) if len(g) >= min_points])
    return centers, bins


def knots(n):
    return np.r_[np.zeros(4), np.linspace(0, 1, n - 2)[1:-1], np.ones(4)]


def design(uv, nu, nv, span, du=0, dv=0):
    u = BSpline(knots(nu), np.eye(nu), 3)(uv[:, 0], nu=du) / span[0] ** du
    v = BSpline(knots(nv), np.eye(nv), 3)(uv[:, 1], nu=dv) / span[1] ** dv
    return np.einsum("ni,nj->nij", u, v).reshape(len(uv), -1)


def penalty_rows(nu, nv, span):
    """The original 25 by 20 uniform samples, including both endpoints."""
    grid = np.array(np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 20))).reshape(2, -1).T
    return np.vstack(
        [
            design(grid, nu, nv, span, 2, 0),
            2**0.5 * design(grid, nu, nv, span, 1, 1),
            design(grid, nu, nv, span, 0, 2),
        ]
    )


def spatial_folds(bins):
    return ((bins[:, 0] // 5) + 2 * (bins[:, 1] // 5)) % 4


def solve(A, z, L, lam):
    w = np.ones(len(z))
    c = np.zeros(A.shape[1])
    for _ in range(4):
        c = np.linalg.solve(
            (A.T * w) @ A / len(z) + lam * (L.T @ L) / len(L) + np.eye(A.shape[1]) * 1e-9,
            (A.T * w) @ z / len(z),
        )
        res = z - A @ c
        mad = max(0.006, 1.4826 * np.median(np.abs(res - np.median(res))))
        w = np.minimum(1, 1.5 * mad / np.maximum(np.abs(res), 1e-9))
    return c


def select_candidate(records):
    """Smallest net within 10% of minimum p90, then strongest penalty."""
    threshold = min(row["cv_p90"] for row in records) * 1.1
    return sorted(
        [row for row in records if row["cv_p90"] <= threshold],
        key=lambda row: (row["nu"] * row["nv"], -row["lambda"]),
    )[0]
