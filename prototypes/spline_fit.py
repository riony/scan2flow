"""v0.1 robust tensor-product spline fit helpers. Caller must supply chart, anchors and smoothing; validation, physical boundaries and units are not handled."""

import numpy as np
from scipy.interpolate import BSpline


def knots(n, a, b):
    return np.r_[np.full(4, a), np.linspace(a, b, n - 2)[1:-1], np.full(4, b)]


def design(xy, uk, vk, du=0, dv=0):
    nu = len(uk) - 4
    nv = len(vk) - 4
    u = BSpline(uk, np.eye(nu), 3).derivative(du)(xy[:, 0])
    v = BSpline(vk, np.eye(nv), 3).derivative(dv)(xy[:, 1])
    return np.einsum("ni,nj->nij", u, v).reshape(len(xy), -1)


def penalty(nu, nv, span):
    return np.r_[
        np.kron(np.diff(np.eye(nu), 2, axis=0), np.eye(nv)) * (nu - 3) ** 2 / span[0] ** 2,
        np.kron(np.eye(nu), np.diff(np.eye(nv), 2, axis=0)) * (nv - 3) ** 2 / span[1] ** 2,
    ]


def solve(A, z, L, lam):
    w = np.ones(len(z))
    for _ in range(6):
        c = np.linalg.solve(
            A.T @ (w[:, None] * A) + lam * (L.T @ L) + np.eye(A.shape[1]) * 1e-9, A.T @ (w * z)
        )
        res = A @ c - z
        s = max(0.004, 1.4826 * np.median(abs(res - np.median(res))))
        w = np.minimum(1, 1.5 * s / np.maximum(abs(res), 1e-12))
    return c
