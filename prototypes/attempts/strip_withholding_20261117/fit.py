"""Historical strip comparison kernels using the under-integrated penalty.

These are diagnostic prototypes, not a validated recipe for filling gaps.
Only caller-supplied coordinates, retained heights and constraint rows enter.
No observation lookup, inference, saved geometry or experiment driver is present.
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import Delaunay


def design(xy, chart, du=0, dv=0):
    q = (xy - np.asarray(chart["lo"])) / np.asarray(chart["span"])
    ku = np.asarray(chart["u_knots_full"])
    kv = np.asarray(chart["v_knots_full"])
    a = BSpline(ku, np.eye(len(ku) - 4), 3)(q[:, 0], nu=du) / chart["span"][0] ** du
    b = BSpline(kv, np.eye(len(kv) - 4), 3)(q[:, 1], nu=dv) / chart["span"][1] ** dv
    return np.einsum("ni,nj->nij", a, b).reshape(len(q), -1)


def penalty(chart):
    q = np.array(np.meshgrid(np.linspace(0, 1, 45), np.linspace(0, 1, 35))).reshape(2, -1).T
    xy = q * np.asarray(chart["span"]) + np.asarray(chart["lo"])
    L = np.vstack(
        [design(xy, chart, 2, 0), np.sqrt(2) * design(xy, chart, 1, 1), design(xy, chart, 0, 2)]
    )
    return L.T @ L / len(L)


def fit(train_xy, train_z, chart, pen, normal_rows=None, normal_targets=None, lam=1e-5):
    """Only retained heights enter this API. Prior rows are fixed before this call."""
    A = design(train_xy, chart)
    w = np.ones(len(train_z))
    eye = np.eye(A.shape[1]) * 1e-10
    ng = np.zeros_like(pen)
    nr = np.zeros(A.shape[1])
    if normal_rows is not None:
        ng = normal_rows.T @ normal_rows / len(normal_rows)
        nr = normal_rows.T @ normal_targets / len(normal_rows)
    for _ in range(4):
        c = np.linalg.solve(
            (A.T * w) @ A / len(train_z) + lam * pen + eye + ng,
            (A.T * w) @ train_z / len(train_z) + nr,
        )
        res = train_z - A @ c
        mad = max(0.006, 1.4826 * np.median(abs(res - np.median(res))))
        w = np.minimum(1, 1.5 * mad / np.maximum(abs(res), 1e-9))
    return c


def folds_from_xy(xy):
    result = []
    for axis, quantiles, width in [(0, [0.2, 0.4, 0.6, 0.8], 0.44), (1, [0.35, 0.65], 0.33)]:
        for quantile in quantiles:
            center = float(np.quantile(xy[:, axis], quantile))
            d = xy[:, axis] - center
            core = abs(d) <= width / 2
            omit = abs(d) <= width / 2 + 0.11
            left = (d < -width / 2 - 0.11) & (d >= -width / 2 - 0.11 - 0.44)
            right = (d > width / 2 + 0.11) & (d <= width / 2 + 0.11 + 0.44)
            bins = np.quantile(xy[core, 1 - axis], [0, 0.25, 0.5, 0.75, 1])
            brackets = []
            for j in range(4):
                b = (xy[:, 1 - axis] >= bins[j]) & (xy[:, 1 - axis] <= bins[j + 1])
                brackets.append([int((left & b).sum()), int((right & b).sum())])
            eligible = (
                int(core.sum()) >= 100
                and int((~omit).sum()) >= 2000
                and min(left.sum(), right.sum()) >= 50
                and np.min(brackets) >= 5
            )
            result.append(
                dict(
                    name=f"{'u' if axis == 0 else 'v'}{int(quantile * 100)}",
                    axis=axis,
                    quantile=quantile,
                    center=center,
                    width=width,
                    guard=0.11,
                    core_ids=np.flatnonzero(core).tolist(),
                    excluded_ids=np.flatnonzero(omit).tolist(),
                    training_ids=np.flatnonzero(~omit).tolist(),
                    bracketing_counts=brackets,
                    bracket_band_width=0.44,
                    eligible=bool(eligible),
                )
            )
    return result


def make_grid(xy, pitch):
    x = np.arange(np.ceil(xy[:, 0].min() / pitch), np.floor(xy[:, 0].max() / pitch) + 1) * pitch
    y = np.arange(np.ceil(xy[:, 1].min() / pitch), np.floor(xy[:, 1].max() / pitch) + 1) * pitch
    xx, yy = np.meshgrid(x, y)
    p = np.c_[xx.ravel(), yy.ravel()]
    inside = Delaunay(xy).find_simplex(p) >= 0
    idx = np.arange(len(p)).reshape(len(y), len(x))
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel()
    d = idx[1:, 1:].ravel()
    faces = np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])
    faces = faces[inside[faces].all(axis=1)]
    return p, inside, faces
