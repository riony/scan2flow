"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

import hashlib
import json
import pickle
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(source, output):
    S, D = Path(source), Path(output)
    D.mkdir(parents=True, exist_ok=False)
    (D / "inspection").mkdir()

    def save(p, x):
        p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    model = pickle.load(open(S / "regional_model.pkl", "rb"))
    cams = {v["name"]: v for v in model["images"].values()}
    par = np.array(model["cameras"][1]["params"])
    K = np.array([[par[0], 0, par[2]], [0, par[1], par[3]], [0, 0, 1]])
    dist = par[4:]
    traces = json.loads((S / "training_traces.json").read_text())["rows"]
    for n, h in json.loads((S / "frozen_reference.json").read_text())["files"].items():
        assert sha(S / n) == h
    # Robust floor reference plane: deterministic RANSAC only provides initial geometry and weak prior.
    P = np.load(S / "floor_neighborhood_UNSCALED.npz")["xyz"]
    rng = np.random.default_rng(20261026)
    best = []
    for _ in range(1500):
        a, b, c = P[rng.choice(len(P), 3, False)]
        nn = np.cross(b - a, c - a)
        nn /= max(np.linalg.norm(nn), 1e-10)
        ii = np.where(abs((P - a) @ nn) < 0.015)[0]
        if len(ii) > len(best):
            best = ii
    Q = P[best]
    origin = np.median(Q, axis=0)
    for _ in range(8):
        _, _, V = np.linalg.svd(Q - origin, full_matrices=False)
        n = V[-1]
        e = np.abs((P - origin) @ n)
        Q = P[e < 0.025]
        origin = np.mean(Q, axis=0)
    _, _, V = np.linalg.svd(Q - origin, full_matrices=False)
    U = V[0]
    N = V[-1]
    W = np.cross(N, U)
    basis = np.array([U, W, N]).T
    save(
        D / "chart.json",
        {
            "origin": origin.tolist(),
            "basis_columns_u_v_normal": basis.tolist(),
            "plane_inliers": len(Q),
            "input_points": len(P),
            "plane_threshold_sfm": 0.025,
            "plane_absolute_distance_median_sfm": float(np.median(abs((Q - origin) @ N))),
            "role": "Initialization and weak depth prior only, not measured outer boundary",
        },
    )

    def rays(row):
        xy = np.array(row["xy_native"])[row["use_for_fit"]]
        im = cams[row["image"]]
        und = cv2.undistortPoints(xy.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        r = np.c_[und, np.ones(len(und))] @ im["R"]
        c = im["center"]
        depth = ((origin - c) @ N) / (r @ N)
        xyz = c + depth[:, None] * r
        return xy, (xyz - origin) @ basis

    for row in traces:
        xy, c = rays(row)
        row["xy"] = xy
        row["chart"] = c
        row["range"] = [float(c[:, 0].min()), float(c[:, 0].max())]
    save(
        D / "initial_extents.json",
        [
            {
                "segment": r["segment"],
                "time": r["time"],
                "range": r["range"],
                "v_range": [float(r["chart"][:, 1].min()), float(r["chart"][:, 1].max())],
            }
            for r in traces
        ],
    )

    def project(xyz, row):
        im = cams[row["image"]]
        q = xyz @ im["R"].T + im["t"]
        x, y = (q[:, :2] / q[:, 2:]).T
        r2 = x * x + y * y
        k1, k2, p1, p2 = dist
        g = 1 + k1 * r2 + k2 * r2 * r2
        return np.c_[
            par[0] * (x * g + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + par[2],
            par[1] * (y * g + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + par[3],
        ]

    def knots(n):
        return np.r_[np.zeros(4), np.linspace(0, 1, n - 2)[1:-1], np.ones(4)]

    def mat(u, n):
        return BSpline.design_matrix(u, knots(n), 3).toarray()

    def stats(e):
        return {
            "n": len(e),
            "median": float(np.median(e)),
            "p90": float(np.quantile(e, 0.9)),
            "max": float(np.max(e)),
        }

    results = []
    allcandidates = []
    for seg in sorted(set(r["segment"] for r in traces)):
        rows = [r for r in traces if r["segment"] == seg]
        grid = np.linspace(min(r["range"][0] for r in rows), max(r["range"][1] for r in rows), 2000)
        support = np.array([sum(r["range"][0] <= u <= r["range"][1] for r in rows) for u in grid])
        idx = np.where(support >= 3)[0]
        assert len(idx) > 2
        # Longest contiguous component only; disconnected intervals must not be bridged.
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        keep = max(groups, key=len)
        lo, hi = grid[keep[[0, -1]]]
        length = hi - lo
        for j, r in enumerate(rows):
            r["fold"] = j % 3
            r["sample_u"] = (
                np.linspace(max(lo, r["range"][0]), min(hi, r["range"][1]), 80)
                if max(lo, r["range"][0]) < min(hi, r["range"][1])
                else np.array([])
            )
        rows = [r for r in rows if len(r["sample_u"])]
        initial = np.concatenate([r["chart"] for r in rows])
        coef = np.polyfit(initial[:, 0], initial[:, 1], 1)

        def initial_cp(n):
            return np.c_[np.polyval(coef, np.linspace(lo, hi, n)), np.zeros(n)]

        def xyz_at(cp, u):
            vw = mat((u - lo) / length, len(cp)) @ cp
            return origin + np.c_[u, vw] @ basis.T

        def image_errors(cp, r):
            uv = project(xyz_at(cp, r["sample_u"]), r)
            xy = r["xy"]
            x, y = xy.T
            predy = np.interp(uv[:, 0], x, y)
            sl = np.interp(uv[:, 0], x, np.gradient(y, x))
            normal = (uv[:, 1] - predy) / np.sqrt(1 + sl * sl)
            outside = np.maximum(x[0] - uv[:, 0], 0) + np.maximum(uv[:, 0] - x[-1], 0)
            return normal, outside

        def solve(n, w, train):
            init = initial_cp(n)

            def fun(flat, image_only=False):
                cp = flat.reshape(n, 2)
                errs = []
                for r in train:
                    normal, outside = image_errors(cp, r)
                    errs.extend([normal / 4, outside / 4])
                if not image_only:
                    # second derivative of displacement in a normalized chart, with explicit dimensionless weight
                    u = np.linspace(0, 1, 80)
                    d2 = BSpline(knots(n), cp, 3).derivative(2)(u)
                    errs.extend([np.sqrt(w) * d2.ravel() / 0.05 / np.sqrt(len(u)), cp[:, 1] / 0.05])
                return np.concatenate(errs)

            fit = least_squares(
                fun,
                init.ravel(),
                bounds=((init - 0.15).ravel(), (init + 0.15).ravel()),
                loss="soft_l1",
                f_scale=1,
                max_nfev=150,
                ftol=1e-7,
                xtol=1e-8,
                gtol=1e-7,
            )
            return fit, fun

        candidates = []
        for n in [4, 6, 8]:
            for w in [0.3, 3, 30]:
                fold_scores = []
                cvrows = []
                converged = True
                for fold in range(3):
                    train = [r for r in rows if r["fold"] != fold]
                    test = [r for r in rows if r["fold"] == fold]
                    fit, _ = solve(n, w, train)
                    converged &= fit.success
                    scores = []
                    for r in test:
                        e, o = image_errors(fit.x.reshape(n, 2), r)
                        s = stats(np.sqrt(e * e + o * o))
                        cvrows.append({"time": r["time"], "fold": fold, **s})
                        scores.append(s["median"])
                    fold_scores.append(float(np.mean(scores)))
                cand = {
                    "segment": seg,
                    "controls": n,
                    "bending_weight": w,
                    "fold_scores_native_px": fold_scores,
                    "cv_mean": float(np.mean(fold_scores)),
                    "cv_standard_error": float(np.std(fold_scores, ddof=1) / np.sqrt(3)),
                    "converged_all_folds": bool(converged),
                    "per_view_cv": cvrows,
                }
                candidates.append(cand)
                print(seg, n, w, cand["cv_mean"], flush=True)
        eligible = [c for c in candidates if c["converged_all_folds"]]
        best = min(eligible, key=lambda c: c["cv_mean"])
        limit = best["cv_mean"] + best["cv_standard_error"]
        chosen = min(
            [c for c in eligible if c["cv_mean"] <= limit],
            key=lambda c: (c["controls"], -c["bending_weight"]),
        )
        n = chosen["controls"]
        w = chosen["bending_weight"]
        fit, fun = solve(n, w, rows)
        cp = fit.x.reshape(n, 2)
        u = np.linspace(lo, hi, 801)
        xyz = xyz_at(cp, u)
        # Image-only Jacobian excludes smoothness/plane priors: expose geometric conditioning.
        eps = 1e-6
        J = np.column_stack(
            [
                (
                    fun(fit.x + np.eye(len(fit.x))[i] * eps, True)
                    - fun(fit.x - np.eye(len(fit.x))[i] * eps, True)
                )
                / (2 * eps)
                for i in range(len(fit.x))
            ]
        )
        sv = np.linalg.svd(J, compute_uv=False)
        per = []
        for r in rows:
            e, o = image_errors(cp, r)
            s = stats(np.sqrt(e * e + o * o))
            uv = project(xyz_at(cp, r["sample_u"]), r)
            per.append(
                {
                    "time": r["time"],
                    "normal_with_outside_penalty_px": s,
                    "outside_trace_count": int(sum(o > 0)),
                    "initial_supported_u": r["range"],
                }
            )
        init = initial_cp(n)
        d = abs(cp - init)
        res = {
            "segment": seg,
            "u_range": [float(lo), float(hi)],
            "reference_plane_support_minimum": 3,
            "training_views": len(rows),
            "selected": chosen,
            "selection_limit": limit,
            "fit_success": bool(fit.success),
            "fit_message": fit.message,
            "nfev": fit.nfev,
            "max_control_displacement_sfm": float(d.max()),
            "bound_active_controls": int(sum((d > 0.1499).ravel())),
            "image_jacobian_singular_values": sv.tolist(),
            "image_jacobian_condition": float(sv[0] / sv[-1]),
            "training_per_view": per,
            "length_sfm": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
            "z_range_sfm": [
                float((mat((u - lo) / length, n) @ cp)[:, 1].min()),
                float((mat((u - lo) / length, n) @ cp)[:, 1].max()),
            ],
            "control_points_chart_vz": cp.tolist(),
            "knots": knots(n).tolist(),
            "degree": 3,
            "curve_sample_count": len(xyz),
            "disposition": "modelled candidate; held-out evaluation pending",
        }
        save(D / (seg + "_curve.json"), res)
        np.savez(
            D / (seg + "_UNSCALED.npz"),
            xyz=xyz,
            u=u,
            control_points_chart_vz=cp,
            knots=knots(n),
            origin=origin,
            basis=basis,
            u_range=[lo, hi],
        )
        with open(D / (seg + "_UNSCALED.obj"), "w") as f:
            f.write("# Unscaled open curve; training-only modelled candidate\n")
            f.writelines("v %.10g %.10g %.10g\n" % tuple(p) for p in xyz)
            f.write("l " + " ".join(map(str, range(1, len(xyz) + 1))) + "\n")
        results.append(res)
        allcandidates.extend(candidates)
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        vw = mat((u - lo) / length, n) @ cp
        for r in rows:
            ax[0].plot(
                r["chart"][:, 0],
                r["chart"][:, 1],
                ".",
                ms=2,
                alpha=0.5,
                label=str(r["time"]) + " plane-init",
            )
        ax[0].plot(u, vw[:, 0], "k", lw=2, label="fitted curve")
        ax[0].set_ylabel("Chart transverse / SfM")
        ax[0].legend(ncol=3, fontsize=7)
        ax[1].plot(u, vw[:, 1])
        ax[1].axhline(0, color="grey", ls="--")
        ax[1].set_xlabel("Chart longitudinal / SfM")
        ax[1].set_ylabel("Normal from prior plane / SfM")
        fig.suptitle(seg + " — training fit, not validation")
        fig.tight_layout()
        fig.savefig(D / "inspection" / (seg + "_chart.png"), dpi=150)
        plt.close(fig)
    save(D / "candidate_comparison.json", allcandidates)
    save(D / "fit_results.json", results)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fit pre-annotated training curves; no imagery read")
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    main(args.source, args.output)
