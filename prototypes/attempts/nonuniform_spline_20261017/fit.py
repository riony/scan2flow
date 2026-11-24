"""Historical nonuniform spline refit with its original sampled penalty.

The later strip audit found severe under-integration of this penalty for a
withheld strip. This module deliberately preserves that method. Its reference
fidelity did not validate the physical curvature of unsupported regions.
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import ndtr


def knots(n, dim, center, lo, span):
    """65% uniform, 35% Gaussian CDF density with 0.55 chart-unit width."""
    x = np.linspace(0, 1, 5001)
    mu = (center[dim] - lo[dim]) / span[dim]
    sd = 0.55 / span[dim]
    g = ndtr((x - mu) / sd)
    g = (g - g[0]) / (g[-1] - g[0])
    cdf = 0.65 * x + 0.35 * g
    inside = np.interp(np.linspace(0, 1, n - 2)[1:-1], cdf, x)
    return np.r_[np.zeros(4), inside, np.ones(4)]


def design(q, ku, kv, span, du=0, dv=0):
    a = BSpline(ku, np.eye(len(ku) - 4), 3)(q[:, 0], nu=du) / span[0] ** du
    b = BSpline(kv, np.eye(len(kv) - 4), 3)(q[:, 1], nu=dv) / span[1] ** dv
    return np.einsum("ni,nj->nij", a, b).reshape(len(q), -1)


def sampled_penalty(ku, kv, span):
    """Historical 45 by 35 Hessian sampling; not exact per-span integration."""
    grid = np.array(np.meshgrid(np.linspace(0, 1, 45), np.linspace(0, 1, 35))).reshape(2, -1).T
    rows = np.vstack(
        [
            design(grid, ku, kv, span, 2, 0),
            2**0.5 * design(grid, ku, kv, span, 1, 1),
            design(grid, ku, kv, span, 0, 2),
        ]
    )
    return rows.T @ rows / len(rows)


def spatial_folds(xy, lo):
    bins = np.floor((xy - lo) / 0.055).astype(int)
    return ((bins[:, 0] // 5) + 2 * (bins[:, 1] // 5)) % 4


def solve(A, z, pen, lam):
    w = np.ones(len(z))
    eye = np.eye(A.shape[1]) * 1e-10
    for _ in range(4):
        c = np.linalg.solve((A.T * w) @ A / len(z) + lam * pen + eye, (A.T * w) @ z / len(z))
        res = z - A @ c
        mad = max(0.006, 1.4826 * np.median(abs(res - np.median(res))))
        w = np.minimum(1, 1.5 * mad / np.maximum(abs(res), 1e-9))
    return c


def select_candidate(records):
    """Preserve the original within-10%-p90 model selection rule."""
    threshold = min(row["cv_p90"] for row in records) * 1.1
    return sorted(
        [row for row in records if row["cv_p90"] <= threshold],
        key=lambda row: (row["nu"] * row["nv"], -row["lambda"]),
    )[0]
