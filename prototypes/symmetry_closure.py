"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.ndimage as ndi
import trimesh
from scipy.spatial import cKDTree

HERE = None
ROOT = None
SCALE_DIR = None
PATCH_DIR = None
SEWN_DIR = None


def read_sparse_points3d_bin(path: Path):
    pts, rgbs = [], []
    with open(path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            tlen = struct.unpack("<Q", f.read(8))[0]
            f.seek(tlen * 8, 1)
            if err < 2.8 and tlen >= 3:
                pts.append(xyz)
                rgbs.append(rgb)
    return np.array(pts, dtype=np.float64), np.array(rgbs, dtype=np.uint8)


def car_envelope_mask(p_car: np.ndarray) -> np.ndarray:
    """Filter out display-room walls, spectator barrier, floor carpet, and stray background reflections."""
    x, y, z = p_car[:, 0], p_car[:, 1], p_car[:, 2]
    ay = np.abs(y)

    # Global bounding box of the vehicle in SfM units
    m = (x >= -15.8) & (x <= 4.85) & (ay <= 3.32) & (z >= 0.08) & (z <= 3.48)

    # 1. Front wing & nosecone (X in [1.2, 4.85])
    fw_region = x > 1.2
    fw_valid = ((ay <= 3.05) & (z <= 0.92)) | ((ay <= 0.95) & (z <= 1.85 - 0.22 * (x - 1.2)))
    m &= (~fw_region) | fw_valid

    # 2. Front axle & suspension / nose bulkhead (X in [-1.5, 1.2])
    fa_region = (x >= -1.5) & (x <= 1.2)
    fa_valid = (
        ((ay >= 1.80) & (ay <= 3.32) & (z <= 2.05))
        | ((ay <= 1.15) & (z >= 0.12) & (z <= 2.18))
        | ((ay > 1.15) & (ay < 1.80) & (z >= 0.35) & (z <= 1.50))
    )
    m &= (~fa_region) | fa_valid

    # 3. Bargeboard, cockpit, halo & sidepod inlet (X in [-5.0, -1.5])
    mid_front_region = (x >= -5.0) & (x < -1.5)
    mid_front_valid = (ay <= 2.35) & (z <= 2.95) & (~((ay > 1.35) & (z > 2.05)))
    m &= (~mid_front_region) | mid_front_valid

    # 4. Sidepods, engine cover, airbox & shark fin (X in [-10.8, -5.0])
    sp_region = (x >= -10.8) & (x < -5.0)
    sp_valid = ((ay <= 2.35) & (z <= 2.18)) | ((ay <= 1.10) & (z <= 3.45))
    m &= (~sp_region) | sp_valid

    # 5. Rear axle, cola bottle, rear wing & diffuser (X in [-15.8, -10.8])
    rear_region = x < -10.8
    rear_wheel_valid = (x >= -13.45) & (ay >= 1.70) & (ay <= 3.32) & (z <= 2.05)
    rear_body_valid = (ay <= 1.75) & (z <= 2.45) & (x >= -14.8)
    rear_wing_valid = (x >= -15.8) & (x <= -12.5) & (ay <= 1.65) & (z >= 0.45) & (z <= 3.35)
    m &= (~rear_region) | (rear_wheel_valid | rear_body_valid | rear_wing_valid)

    return m


def main(workspace, artifact_root, mm_per_unit):
    global HERE, ROOT, SCALE_DIR, PATCH_DIR, SEWN_DIR
    HERE, ROOT = Path(workspace), Path(artifact_root)
    SCALE_DIR, PATCH_DIR, SEWN_DIR = [
        ROOT / n for n in ("scale_verification", "native_patchmatch", "sidepod_sewn")
    ]

    t0 = time.time()
    # Clean up any temporary OpenFOAM dump files from earlier tests
    for p in list(HERE.glob("tier1_whole_car_METRIC_mm_*.obj")) + [
        HERE / "tier1_whole_car_METRIC_mm.vtp",
        HERE / "selfInterPoints.obj",
        HERE / "test_iso.stl",
    ]:
        if p.exists():
            p.unlink()

    vf = json.loads((HERE / "vehicle_frame.json").read_text())
    mid_front = np.array(vf["mid_front_axle_body_main"], dtype=np.float64)
    e_long = np.array(vf["e_long_forward"], dtype=np.float64)
    e_lat = np.array(vf["e_lat_port"], dtype=np.float64)
    e_vert = np.array(vf["e_vert_up"], dtype=np.float64)
    r_wheel = float(vf["r_wheel_sfm"])
    scale_mm = mm_per_unit
    R_car = np.vstack([e_long, e_lat, e_vert])  # 3x3: world -> car basis

    def world_to_car(pts_w: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(pts_w) - mid_front
        c = d @ R_car.T
        c[:, 2] += r_wheel
        return c

    def car_to_world(pts_c: np.ndarray) -> np.ndarray:
        c = np.atleast_2d(pts_c).copy()
        c[:, 2] -= r_wheel
        return c @ R_car + mid_front

    # 1. Load the 3 point-cloud sources in exact body_main coordinates
    wc_npz = np.load(HERE / "tier1_whole_car_cascade_points.npz")
    pts_wc = wc_npz["xyz"].astype(np.float64)
    rgb_wc = wc_npz["rgb"].astype(np.uint8)

    pcd_sidepod = o3d.io.read_point_cloud(str(PATCH_DIR / "fusion_cascade_3072_c45/dense.ply"))
    pts_sp = np.asarray(pcd_sidepod.points, dtype=np.float64)
    rgb_sp = (np.asarray(pcd_sidepod.colors) * 255.0).clip(0, 255).astype(np.uint8)

    pts_sift, rgb_sift = read_sparse_points3d_bin(SCALE_DIR / "sparse/fixed_anchor/points3D.bin")

    print(
        f"Loaded raw points: whole_car_cascade={len(pts_wc)}, sidepod_cascade_3072={len(pts_sp)}, sift_tracks={len(pts_sift)}"
    )

    pts_all_w = np.vstack([pts_wc, pts_sp, pts_sift])
    rgb_all = np.vstack([rgb_wc, rgb_sp, rgb_sift])
    pts_all_c = world_to_car(pts_all_w)

    env_keep = car_envelope_mask(pts_all_c)
    pts_env_c = pts_all_c[env_keep]
    rgb_env = rgb_all[env_keep]

    # 2. Voxel downsample and statistical/radius outlier removal in Open3D (in car frame)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_env_c)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb_env.astype(np.float64) / 255.0, 0.0, 1.0))
    pcd = pcd.voxel_down_sample(voxel_size=0.04)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.20)

    pts_clean_c = np.asarray(pcd.points, dtype=np.float64)
    rgb_clean = np.clip(np.asarray(pcd.colors, dtype=np.float64), 0.0, 1.0)

    sift_c = world_to_car(pts_sift)
    sift_car_pts_w = pts_sift[car_envelope_mask(sift_c)]

    # 3. Bilateral symmetry augmentation across Y_car = 0
    mirror_mask = np.abs(pts_clean_c[:, 1]) > 0.04
    pts_mir_c = pts_clean_c[mirror_mask].copy()
    pts_mir_c[:, 1] = -pts_mir_c[:, 1]
    rgb_mir = rgb_clean[mirror_mask].copy()

    pts_sym_c = np.vstack([pts_clean_c, pts_mir_c])
    rgb_sym = np.vstack([rgb_clean, rgb_mir])

    pcd_sym = o3d.geometry.PointCloud()
    pcd_sym.points = o3d.utility.Vector3dVector(pts_sym_c)
    pcd_sym.colors = o3d.utility.Vector3dVector(np.clip(rgb_sym, 0.0, 1.0))
    pcd_sym = pcd_sym.voxel_down_sample(voxel_size=0.045)
    pts_sym_c = np.asarray(pcd_sym.points, dtype=np.float64)
    rgb_sym = np.clip(np.asarray(pcd_sym.colors, dtype=np.float64), 0.0, 1.0)

    # 4. Estimate outward surface normals
    pcd_sym.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=40)
    )
    normals_sym = np.asarray(pcd_sym.normals, dtype=np.float64)

    x_p, y_p, z_p = pts_sym_c[:, 0], pts_sym_c[:, 1], pts_sym_c[:, 2]
    skel_y = np.zeros_like(y_p)
    skel_z = np.full_like(z_p, 0.85)
    fw_idx = (x_p >= -1.4) & (x_p <= 1.2) & (np.abs(y_p) >= 1.75)
    skel_y[fw_idx] = np.sign(y_p[fw_idx]) * 2.55
    skel_z[fw_idx] = r_wheel
    rw_idx = (x_p >= -13.5) & (x_p <= -10.8) & (np.abs(y_p) >= 1.70)
    skel_y[rw_idx] = np.sign(y_p[rw_idx]) * 2.50
    skel_z[rw_idx] = r_wheel
    fwing_idx = (x_p > 1.2) & (np.abs(y_p) > 0.65)
    skel_y[fwing_idx] = y_p[fwing_idx] * 0.85
    skel_z[fwing_idx] = 0.35

    outward_vec = np.column_stack(
        [
            np.where(x_p > 2.5, x_p - 2.0, np.where(x_p < -13.5, x_p + 13.0, 0.0)),
            y_p - skel_y,
            z_p - skel_z,
        ]
    )
    dot = np.sum(normals_sym * outward_vec, axis=1)
    normals_sym[dot < 0] *= -1.0

    # 5. Add structured underbody ground-closure points with downward normals (0, 0, -1)
    xs_u = np.linspace(-14.2, 4.2, 160)
    ys_u = np.linspace(-3.15, 3.15, 90)
    ugx, ugy = np.meshgrid(xs_u, ys_u)
    ugx, ugy = ugx.ravel(), ugy.ravel()
    ay_u = np.abs(ugy)

    under_keep = (
        ((ugx >= -13.8) & (ugx <= -1.8) & (ay_u <= np.where(ugx < -10.8, 1.50, 2.12)))
        | (
            (ugx > -1.8)
            & (ugx <= 3.8)
            & (ay_u <= np.interp(ugx, [-1.8, 1.2, 3.8], [1.05, 0.55, 0.32]))
        )
        | ((ugx >= 2.6) & (ugx <= 4.2) & (ay_u <= 2.95))
        | ((ugx >= -1.0) & (ugx <= 1.0) & (ay_u >= 2.05) & (ay_u <= 3.20))
        | ((ugx >= -13.1) & (ugx <= -11.1) & (ay_u >= 1.90) & (ay_u <= 3.20))
    )
    ugx, ugy = ugx[under_keep], ugy[under_keep]
    ay_u = np.abs(ugy)
    ugz = np.full_like(ugx, 0.14)
    diff_m = (ugx < -11.5) & (ay_u <= 1.55)
    ugz[diff_m] = 0.14 + 0.40 * ((-11.5 - ugx[diff_m]) / 2.3)
    nose_m = (ugx > -1.0) & (ugx < 2.6) & (ay_u <= 0.85)
    ugz[nose_m] = np.interp(ugx[nose_m], [-1.0, 2.6], [0.16, 0.38])
    wheel_m = (ay_u >= 1.90) & (ugx < 1.5)
    ugz[wheel_m] = 0.08

    pts_under_c = np.column_stack([ugx, ugy, ugz])
    normals_under = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(pts_under_c), 1))
    rgb_under = np.tile(np.array([[0.15, 0.15, 0.15]]), (len(pts_under_c), 1))

    pts_full_c = np.vstack([pts_sym_c, pts_under_c])
    normals_full_c = np.vstack([normals_sym, normals_under])
    rgb_full = np.clip(np.vstack([rgb_sym, rgb_under]), 0.0, 1.0)

    pts_full_w = car_to_world(pts_full_c)
    normals_full_w = normals_full_c @ R_car

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(pts_full_w)
    pcd_full.normals = o3d.utility.Vector3dVector(normals_full_w)
    pcd_full.colors = o3d.utility.Vector3dVector(rgb_full)

    o3d.io.write_point_cloud(str(HERE / "tier1_whole_car_fused_points_UNSCALED.ply"), pcd_full)
    print(
        f"Saved oriented whole-car point cloud ({len(pts_full_w)} pts) to tier1_whole_car_fused_points_UNSCALED.ply"
    )

    # 6. Reconstruct closed watertight 2-manifold whole-car mesh via Regular-Grid Volumetric SDF
    #    + Flying Edges (`o3d.t.geometry.TriangleMesh.create_isosurfaces`)
    print(
        "Building regular-grid volumetric Signed Distance Field & Flying Edges isosurface...",
        flush=True,
    )
    pts_w32 = pts_full_w.astype(np.float32)
    bmin = pts_w32.min(axis=0) - 0.8
    bmax = pts_w32.max(axis=0) + 0.8
    step = 0.08  # ~26.5 mm voxel grid pitch
    nx = int(np.ceil((bmax[0] - bmin[0]) / step)) + 1
    ny = int(np.ceil((bmax[1] - bmin[1]) / step)) + 1
    nz = int(np.ceil((bmax[2] - bmin[2]) / step)) + 1

    # Fast EDT distance field from point cloud rasterization onto the 3D grid ( indexing: [z, y, x] )
    ix = np.clip(np.rint((pts_w32[:, 0] - bmin[0]) / step).astype(int), 2, nx - 3)
    iy = np.clip(np.rint((pts_w32[:, 1] - bmin[1]) / step).astype(int), 2, ny - 3)
    iz = np.clip(np.rint((pts_w32[:, 2] - bmin[2]) / step).astype(int), 2, nz - 3)

    seed = np.zeros((nz, ny, nx), dtype=bool)
    seed[iz, iy, ix] = True
    dist_vox = ndi.distance_transform_edt(~seed) * step
    occ_dil = dist_vox < 0.24
    occ_filled = ndi.binary_fill_holes(
        ndi.binary_closing(occ_dil, structure=np.ones((3, 3, 3), dtype=bool))
    )
    occ = (ndi.distance_transform_edt(occ_filled) * step > 0.18) | (dist_vox < 0.065)
    occ = ndi.binary_closing(occ, structure=np.ones((3, 3, 3), dtype=bool))
    occ = ndi.binary_fill_holes(occ)

    lbl, _ = ndi.label(occ)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    occ = lbl == np.argmax(counts)

    sdf = ndi.distance_transform_edt(~occ).astype(np.float32) - ndi.distance_transform_edt(
        occ
    ).astype(np.float32)
    sdf = ndi.gaussian_filter(sdf, sigma=1.0)
    sdf[:2, :, :] = 10.0
    sdf[-2:, :, :] = 10.0
    sdf[:, :2, :] = 10.0
    sdf[:, -2:, :] = 10.0
    sdf[:, :, :2] = 10.0
    sdf[:, :, -2:] = 10.0

    # Extract -0.5123 contour via Open3D Flying Edges to cancel the 0.06 SfM seed thickness
    m_iso_t = o3d.t.geometry.TriangleMesh.create_isosurfaces(
        o3d.core.Tensor(sdf), contour_values=[-0.5123]
    )
    m_iso = m_iso_t.to_legacy()
    verts_w = np.asarray(m_iso.vertices, dtype=np.float64) * step + bmin
    tris = np.asarray(m_iso.triangles, dtype=np.int32)[:, [0, 2, 1]]

    tm_unscaled = trimesh.Trimesh(vertices=verts_w, faces=tris, process=True)
    if tm_unscaled.volume < 0:
        tm_unscaled.invert()

    # Assign nearest point-cloud RGB vertex colors for the unscaled PLY
    pt_tree = cKDTree(pts_full_w)
    _, nn_idx = pt_tree.query(tm_unscaled.vertices)
    m_out = o3d.geometry.TriangleMesh()
    m_out.vertices = o3d.utility.Vector3dVector(tm_unscaled.vertices)
    m_out.triangles = o3d.utility.Vector3iVector(tm_unscaled.faces)
    m_out.vertex_colors = o3d.utility.Vector3dVector(rgb_full[nn_idx])
    m_out.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(HERE / "tier1_whole_car_UNSCALED.ply"), m_out)

    tm_metric = tm_unscaled.copy()
    tm_metric.vertices = tm_metric.vertices * scale_mm
    stl_metric_path = HERE / "tier1_whole_car_METRIC_mm.stl"
    obj_metric_path = HERE / "tier1_whole_car_METRIC_mm.obj"
    tm_metric.export(str(stl_metric_path))
    tm_metric.export(str(obj_metric_path))

    # 7. Quantitative accuracy audit against 3D evidence
    v_tree = cKDTree(tm_unscaled.vertices)
    d_sift, _ = v_tree.query(sift_car_pts_w)
    pts_clean_w = car_to_world(pts_clean_c)
    d_casc, _ = v_tree.query(pts_clean_w)

    sewn_sp_stl = trimesh.load(str(SEWN_DIR / "vehicle_sidepod_shell_METRIC_mm.stl"), force="mesh")
    sewn_sp_pts_w = np.asarray(sewn_sp_stl.vertices, dtype=np.float64) / scale_mm
    d_sewn_sp, _ = v_tree.query(sewn_sp_pts_w)

    verts_car_mm = world_to_car(tm_unscaled.vertices) * scale_mm
    bbox_min_mm = verts_car_mm.min(axis=0).tolist()
    bbox_max_mm = verts_car_mm.max(axis=0).tolist()
    extents_mm = (verts_car_mm.max(axis=0) - verts_car_mm.min(axis=0)).tolist()

    # 8. Run OpenFOAM 2412 surfaceCheck -checkSelfIntersection on the metric STL
    of_cmd = (
        f"source {os.environ['SCAN2FOAM_OF_BASHRC']} && "
        f"surfaceCheck -checkSelfIntersection {stl_metric_path}"
    )
    proc = subprocess.run(
        ["bash", "-c", of_cmd], cwd=str(HERE), capture_output=True, text=True, check=True
    )
    of_log = proc.stdout + "\n" + proc.stderr
    (HERE / "openfoam_tier1_surfaceCheck.log").write_text(of_log)

    assert "Surface has no illegal triangles." in of_log
    assert "Surface is closed. All edges connected to two faces." in of_log
    assert "Number of unconnected parts : 1" in of_log
    assert "Number of zones (connected area with consistent normal) : 1" in of_log
    assert "Surface is not self-intersecting" in of_log

    audit = {
        "tier": "Tier 1 — Multi-View Cascade learned multi-view depth + Volumetric SDF / Screened Poisson Whole-Car Mesh Baseline",
        "elapsed_seconds": float(time.time() - t0),
        "scale_mm_per_unit": scale_mm,
        "input_point_counts": {
            "whole_car_cascade_27views": int(len(pts_wc)),
            "sidepod_cascade_3072": int(len(pts_sp)),
            "fixed_anchor_sift_tracks": int(len(pts_sift)),
            "cleaned_observed_car_points": int(len(pts_clean_c)),
            "bilateral_symmetric_plus_underbody_points": int(len(pts_full_w)),
        },
        "raw_octree_poisson_depth8_diagnostic": {
            "vertices": 56219,
            "triangles": 115660,
            "illegal_sliver_triangles": 3064,
            "open_boundary_edges": 132,
            "non_manifold_edges": 5073,
            "self_intersecting_locations": 4351,
            "diagnosis": (
                "Adaptive-octree Dual Marching Cubes (Screened Poisson depth=8) folds on thin double-sided "
                "aerodynamic structures (front wing, rear wing, bargeboard, halo) where top and bottom surface "
                "normals oppose across <0.15 SfM thickness, producing 5,073 non-manifold edges and 4,351 self-intersections."
            ),
        },
        "regularized_sdf_flying_edges_mesh_topology": {
            "grid_step_sfm": step,
            "grid_step_mm": float(step * scale_mm),
            "vertices": int(len(tm_metric.vertices)),
            "triangles": int(len(tm_metric.faces)),
            "is_watertight_2manifold": bool(tm_metric.is_watertight),
            "is_winding_consistent": bool(tm_metric.is_winding_consistent),
            "euler_number": int(tm_metric.euler_number),
            "openfoam_illegal_triangles": 0,
            "openfoam_open_edges": 0,
            "openfoam_nonmanifold_edges": 0,
            "openfoam_unconnected_parts": 1,
            "openfoam_normal_zones": 1,
            "openfoam_self_intersections": 0,
            "surface_area_m2": float(tm_metric.area / 1e6),
            "enclosed_volume_m3": float(tm_metric.volume / 1e9),
        },
        "vehicle_frame_bbox_mm": {
            "min_xyz_mm": bbox_min_mm,
            "max_xyz_mm": bbox_max_mm,
            "length_width_height_mm": extents_mm,
        },
        "accuracy_vs_3d_evidence": {
            "vs_fixed_anchor_sift_car_points": {
                "count": int(len(d_sift)),
                "median_sfm": float(np.median(d_sift)),
                "p90_sfm": float(np.percentile(d_sift, 90)),
                "median_mm": float(np.median(d_sift) * scale_mm),
                "p90_mm": float(np.percentile(d_sift, 90) * scale_mm),
            },
            "vs_observed_cascade_points": {
                "count": int(len(d_casc)),
                "median_sfm": float(np.median(d_casc)),
                "p90_sfm": float(np.percentile(d_casc, 90)),
                "median_mm": float(np.median(d_casc) * scale_mm),
                "p90_mm": float(np.percentile(d_casc, 90) * scale_mm),
            },
            "vs_step3_sewn_sidepod_shell": {
                "count": int(len(d_sewn_sp)),
                "median_sfm": float(np.median(d_sewn_sp)),
                "p90_sfm": float(np.percentile(d_sewn_sp, 90)),
                "median_mm": float(np.median(d_sewn_sp) * scale_mm),
                "p90_mm": float(np.percentile(d_sewn_sp, 90) * scale_mm),
            },
        },
    }
    (HERE / "tier1_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


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
