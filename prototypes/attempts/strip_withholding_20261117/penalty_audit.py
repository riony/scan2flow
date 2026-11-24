"""Evaluate supplied coefficients without fitting or selecting a new surface.

Four Gauss-Legendre sites in every bicubic knot span integrate the squared
Hessian polynomials, up to roundoff. This diagnoses the old sampling objective;
it is not a corrected fitting implementation or evidence that a refit succeeds.
"""

import numpy as np

from .fit import design


def audit_energies(chart, coefficients):
    """Return energy arrays for one vector or columns of supplied coefficients.

    Both energies average [h_uu, sqrt(2) h_uv, h_vv] over normalized chart
    area and divide by three. Physical derivative scaling comes from design.
    No input coefficients are changed.
    """
    C = np.asarray(coefficients)
    if C.ndim == 1:
        C = C[:, None]
    x, w = np.polynomial.legendre.leggauss(4)
    grids, weights, support = [], [], {}
    for axis, n in [("u", 45), ("v", 35)]:
        t = np.unique(chart[axis + "_knots_full"])
        g = np.linspace(0, 1, n)
        empty = []
        for lo, hi in zip(t[:-1], t[1:]):
            if not np.any((g > lo) & (g < hi)):
                empty.append([float(lo), float(hi)])
        support[axis] = {
            "interval_count": len(t) - 1,
            "intervals_without_interior_uniform_penalty_site": empty,
        }
        grids.append(
            np.concatenate([(lo + hi) / 2 + (hi - lo) / 2 * x for lo, hi in zip(t[:-1], t[1:])])
        )
        weights.append(np.concatenate([(hi - lo) / 2 * w for lo, hi in zip(t[:-1], t[1:])]))
    uv = np.array(np.meshgrid(*grids)).reshape(2, -1).T
    weight = np.outer(weights[1], weights[0]).ravel()
    xy = uv * np.asarray(chart["span"]) + np.asarray(chart["lo"])
    uniform = np.array(np.meshgrid(np.linspace(0, 1, 45), np.linspace(0, 1, 35))).reshape(2, -1).T
    uniform = uniform * np.asarray(chart["span"]) + np.asarray(chart["lo"])
    exact = np.zeros(C.shape[1])
    sampled = np.zeros(C.shape[1])
    for du, dv, factor in [(2, 0, 1), (1, 1, 2), (0, 2, 1)]:
        values = design(xy, chart, du, dv) @ C
        exact += factor * np.sum(weight[:, None] * values**2, axis=0) / 3
        values = design(uniform, chart, du, dv) @ C
        sampled += factor * np.mean(values**2, axis=0) / 3
    return {
        "basis_support": support,
        "historic_uniform_sampled_energy": sampled,
        "per_span_integrated_energy": exact,
    }
