"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

import json
import struct
from pathlib import Path

import numpy as np

HERE = None
ROOT = None
SCALE_DIR = None
SEWN_DIR = None
PATCH_DIR = None


def read_sparse_points3d_bin(path: Path):
    pts, rgbs, errs, tlens = [], [], [], []
    with open(path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            tlen = struct.unpack("<Q", f.read(8))[0]
            f.seek(tlen * 8, 1)
            if err < 3.5 and tlen >= 3:
                pts.append(xyz)
                rgbs.append(rgb)
                errs.append(err)
                tlens.append(tlen)
    return (
        np.array(pts, dtype=float),
        np.array(rgbs, dtype=np.uint8),
        np.array(errs),
        np.array(tlens),
    )


def main(workspace, artifact_root, mm_per_unit):
    global HERE, ROOT, SCALE_DIR, PATCH_DIR, SEWN_DIR
    HERE, ROOT = Path(workspace), Path(artifact_root)
    SCALE_DIR, PATCH_DIR, SEWN_DIR = [
        ROOT / n for n in ("scale_verification", "native_patchmatch", "sidepod_sewn")
    ]

    rear_m = json.loads((SCALE_DIR / "rear_wheel_metrics.json").read_text())
    front_m = json.loads((SCALE_DIR / "front_wheel_metrics.json").read_text())
    free_bounds = json.loads((SEWN_DIR / "free_boundaries.json").read_text())

    scale_mm = mm_per_unit
    r_wheel = float(rear_m["outer_tread_cylinder"]["coaxial_radius_sfm"])  # 0.994212 SfM = 330.0 mm

    # Note on wheel labels:
    # In 74-86s (body_main), the camera walks from the rear wheel to the cockpit on the RIGHT (starboard) side of the car!
    # In scale_verification, the 74-86s rear wheel was called "rear_left" in JSON keys,
    # the 88-96s/34-42s front wheel on the SAME side was called "front_left",
    # and the opposite-side front wheel (seen in 47-50s and 101-110s) was called "front_right".
    # In right-handed vehicle coordinates (+X forward, +Z upward, +Y port/left, -Y starboard/right):
    c_rear_starboard = np.array(
        rear_m["sidewall_plane"]["wheel_center_outer_plane_body_main"], dtype=float
    )
    c_front_starboard = np.array(
        front_m["front_left_wheel"]["sparse_exact_intrinsics_crosscheck"]["wheel_center_body_main"],
        dtype=float,
    )
    c_front_port = np.array(
        front_m["front_right_wheel"]["wheel_center_outer_plane_body_main"], dtype=float
    )

    # Upward normal to the ground plane from the 3 wheel centers
    e_vert = np.cross(c_front_starboard - c_rear_starboard, c_front_port - c_front_starboard)
    e_vert /= np.linalg.norm(e_vert)
    if e_vert[1] > 0:
        e_vert = -e_vert

    # Forward longitudinal axis (+X_car) along starboard wheelbase projected onto ground plane
    e_long = (c_front_starboard - c_rear_starboard) - np.dot(
        c_front_starboard - c_rear_starboard, e_vert
    ) * e_vert
    e_long /= np.linalg.norm(e_long)

    # Lateral axis (+Y_car pointing toward port/left side, so e_long x e_lat = e_vert => e_lat = e_vert x e_long)
    e_lat = np.cross(e_vert, e_long)
    e_lat /= np.linalg.norm(e_lat)

    # Let's check the lateral coordinate of the 4 upper skin outer edges (which lie along the dorsal spine / centerline)
    # and the midpoint of the front wheels:
    mid_front = 0.5 * (c_front_starboard + c_front_port)
    # Wait: let's check where the upper_skin_outer_* edges lie relative to mid_front along e_lat:
    upper_outer_pts = []
    for fb in free_bounds:
        if fb["name"].startswith("upper_skin_outer_"):
            upper_outer_pts.extend(
                [fb["start_xyz_unscaled"], fb["midpoint_xyz_unscaled"], fb["end_xyz_unscaled"]]
            )
    upper_outer_pts = np.array(upper_outer_pts, dtype=float)
    y_upper_rel_midfront = (upper_outer_pts - mid_front) @ e_lat
    print(
        "Upper skin outer edges Y relative to mid_front (SfM):", np.round(y_upper_rel_midfront, 4)
    )
    print(
        "Median Y of upper_skin_outer_* relative to mid_front:",
        float(np.median(y_upper_rel_midfront)),
    )

    # Let's check why upper_skin_outer_* has Y ≈ +0.13 SfM (+43 mm) relative to mid_front:
    # Because upper_skin_outer_* is on the starboard sidepod (-Y side), and its width w(x) was extended slightly across the spine crest,
    # OR let's check where the actual dorsal spine / T-cam / nosecone centerline points in sparse reconstruction lie!
    pts_sparse, rgbs_sparse, errs_sparse, tlens_sparse = read_sparse_points3d_bin(
        SCALE_DIR / "sparse/fixed_anchor/points3D.bin"
    )

    def to_car_raw(pts):
        d = np.atleast_2d(pts) - mid_front
        return np.column_stack([d @ e_long, d @ e_lat, d @ e_vert + r_wheel])

    p_car = to_car_raw(pts_sparse)

    # Check highest points (T-cam / airbox / shark fin) in X in [-10.5, -5.5], Z > 2.8
    high_mask = (
        (p_car[:, 0] >= -10.5)
        & (p_car[:, 0] <= -5.5)
        & (p_car[:, 2] >= 2.85)
        & (np.abs(p_car[:, 1]) < 0.8)
    )
    print(
        f"High spine/airbox/T-cam points ({high_mask.sum()} pts): median Y = {np.median(p_car[high_mask, 1]):.4f} SfM"
    )

    # Check nosecone centerline in X in [0.0, 4.2], |Y| < 1.2, Z in [0.5, 2.2]
    for x0, x1 in [(-2.0, 0.0), (0.0, 1.5), (1.5, 3.0), (3.0, 4.3)]:
        m = (
            (p_car[:, 0] >= x0)
            & (p_car[:, 0] < x1)
            & (np.abs(p_car[:, 1]) < 1.0)
            & (p_car[:, 2] > 0.4)
            & (p_car[:, 2] < 2.3)
        )
        if m.sum() > 0:
            print(
                f"Nose/chassis X in [{x0:4.1f}, {x1:4.1f}] ({m.sum():4d} pts): Y_p10={np.percentile(p_car[m, 1], 10):6.3f}, Y_med={np.median(p_car[m, 1]):6.3f}, Y_p90={np.percentile(p_car[m, 1], 90):6.3f}, Z_p10={np.percentile(p_car[m, 2], 10):5.3f}, Z_p90={np.percentile(p_car[m, 2], 90):5.3f}"
            )

    # Check front wing span in X in [2.5, 4.6], Z in [0.05, 0.95]
    fw_mask = (
        (p_car[:, 0] >= 2.5)
        & (p_car[:, 0] <= 4.6)
        & (np.abs(p_car[:, 1]) < 3.3)
        & (p_car[:, 2] >= 0.05)
        & (p_car[:, 2] <= 0.95)
    )
    print(f"Front wing points ({fw_mask.sum()} pts):")
    print(
        f"  X: p05={np.percentile(p_car[fw_mask, 0], 5):.3f}, p50={np.median(p_car[fw_mask, 0]):.3f}, p95={np.percentile(p_car[fw_mask, 0], 95):.3f}"
    )
    print(
        f"  Y: p02={np.percentile(p_car[fw_mask, 1], 2):.3f}, p50={np.median(p_car[fw_mask, 1]):.3f}, p98={np.percentile(p_car[fw_mask, 1], 98):.3f}"
    )
    print(
        f"  Z: p05={np.percentile(p_car[fw_mask, 2], 5):.3f}, p50={np.median(p_car[fw_mask, 2]):.3f}, p95={np.percentile(p_car[fw_mask, 2], 95):.3f}"
    )

    # Save canonical vehicle frame definition to JSON so all Tier 1, Tier 2, Tier 3 scripts share the exact frame!
    # Notice: if we set the symmetry plane Y_sym at the exact dorsal spine / nosecone / front-axle symmetry plane,
    # let's check what Y_sym gives symmetric front wing & nosecone & spine:
    y_sym_offset = 0.0  # mid_front has FL at Y = -3.2759 and FR at Y = +3.2759, and nose X in [1.5, 4.3] median Y ~ 0!
    car_frame = {
        "origin_body_main": (mid_front - r_wheel * e_vert).tolist(),
        "mid_front_axle_body_main": mid_front.tolist(),
        "e_long_forward": e_long.tolist(),
        "e_lat_port": e_lat.tolist(),
        "e_vert_up": e_vert.tolist(),
        "r_wheel_sfm": r_wheel,
        "scale_mm_per_unit": scale_mm,
        "wheels_car_frame": {
            "rear_starboard_outer": to_car_raw(c_rear_starboard)[0].tolist(),
            "front_starboard_outer": to_car_raw(c_front_starboard)[0].tolist(),
            "front_port_outer": to_car_raw(c_front_port)[0].tolist(),
            "rear_port_outer": [
                float(to_car_raw(c_rear_starboard)[0][0]),
                float(-to_car_raw(c_rear_starboard)[0][1]),
                float(to_car_raw(c_rear_starboard)[0][2]),
            ],
        },
    }
    (HERE / "vehicle_frame.json").write_text(json.dumps(car_frame, indent=2))
    print("\nSaved vehicle_frame.json")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Legacy recipe; see limitations before running")
    ap.add_argument("--workspace", required=True, type=Path)
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("--mm-per-unit", required=True, type=float)
    args = ap.parse_args()
    if args.mm_per_unit <= 0:
        ap.error("mm-per-unit must be positive and independently justified")
    main(args.workspace, args.artifact_root, args.mm_per_unit)
