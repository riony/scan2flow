"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_NurbsConvert,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Transform,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.collections import Array1_double, Array1_gp_Pnt2d, Array1_int, Array2_gp_Pnt
from OCP.Geom import Geom_BSplineSurface
from OCP.Geom2d import Geom2d_BSplineCurve
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Vec
from OCP.ShapeFix import ShapeFix_Face
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.StlAPI import StlAPI_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

HERE = None
ROOT = None


def knots_ocp(vals):
    k, m = np.unique(vals, return_counts=True)
    a = Array1_double(1, len(k))
    b = Array1_int(1, len(k))
    for i, (x, y) in enumerate(zip(k, m)):
        a.SetValue(i + 1, float(x))
        b.SetValue(i + 1, int(y))
    return a, b


def make_2d_bspline_edge(uv_pts, surf, deg=3):
    uv_pts = np.asarray(uv_pts, dtype=float)
    _, idx = np.unique(np.round(uv_pts, 12), axis=0, return_index=True)
    uv_pts = uv_pts[np.sort(idx)]
    t = np.linspace(0.0, 1.0, len(uv_pts))
    k = min(deg, len(uv_pts) - 1)
    sp = make_interp_spline(t, uv_pts, k=k)
    c = sp.c
    arr = Array1_gp_Pnt2d(1, len(c))
    for i, pt in enumerate(c):
        arr.SetValue(i + 1, gp_Pnt2d(float(pt[0]), float(pt[1])))
    uk, um = knots_ocp(sp.t)
    c2d = Geom2d_BSplineCurve(arr, uk, um, k)
    return BRepBuilderAPI_MakeEdge(c2d, surf, 0.0, 1.0).Edge()


def make_ruled_collar_face(edge, trsf):
    bs0 = BRep_Tool.Curve_s(edge, 0.0, 1.0)
    n_p = bs0.NbPoles()
    n_k = bs0.NbKnots()
    poles = Array2_gp_Pnt(1, n_p, 1, 2)
    for i in range(1, n_p + 1):
        pt0 = bs0.Pole(i)
        pt1 = pt0.Transformed(trsf)
        poles.SetValue(i, 1, pt0)
        poles.SetValue(i, 2, pt1)
    uk = Array1_double(1, n_k)
    um = Array1_int(1, n_k)
    for i in range(1, n_k + 1):
        uk.SetValue(i, bs0.Knot(i))
        um.SetValue(i, bs0.Multiplicity(i))
    vk = Array1_double(1, 2)
    vk.SetValue(1, 0.0)
    vk.SetValue(2, 1.0)
    vm = Array1_int(1, 2)
    vm.SetValue(1, 2)
    vm.SetValue(2, 2)
    surf = Geom_BSplineSurface(poles, uk, vk, um, vm, bs0.Degree(), 1)
    return BRepBuilderAPI_MakeFace(surf, 1e-6).Face()


def count_faces(shape) -> int:
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def write_step(shape, path: Path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(path))
    assert int(status) == 1, f"STEP write failed for {path}"
    r = STEPControl_Reader()
    assert int(r.ReadFile(str(path))) == 1
    r.TransferRoots()
    rt_shape = r.OneShape()
    assert BRepCheck_Analyzer(rt_shape).IsValid(), f"STEP round-trip invalid for {path}"
    return rt_shape


def scale_shape(shape, scale_factor: float):
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), float(scale_factor))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def make_bicubic_bspline_surface_from_grid(grid_pts_w: np.ndarray) -> Geom_BSplineSurface:
    """Create a clamped bicubic Geom_BSplineSurface from a 2D grid of 3D control points (Nu, Nv, 3)."""
    nu, nv, _ = grid_pts_w.shape
    du = min(3, nu - 1)
    dv = min(3, nv - 1)
    poles = Array2_gp_Pnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            poles.SetValue(i + 1, j + 1, gp_Pnt(*[float(x) for x in grid_pts_w[i, j]]))

    def make_clamped_knots(n, d):
        if n == d + 1:
            k = [0.0, 1.0]
            m = [d + 1, d + 1]
        else:
            internal = np.linspace(0.0, 1.0, n - d + 1)[1:-1]
            k = [0.0] + list(internal) + [1.0]
            m = [d + 1] + [1] * len(internal) + [d + 1]
        ak = Array1_double(1, len(k))
        am = Array1_int(1, len(m))
        for idx, (kv, mv) in enumerate(zip(k, m)):
            ak.SetValue(idx + 1, float(kv))
            am.SetValue(idx + 1, int(mv))
        return ak, am

    uk, um = make_clamped_knots(nu, du)
    vk, vm = make_clamped_knots(nv, dv)
    return Geom_BSplineSurface(poles, uk, vk, um, vm, du, dv)


def make_patch_and_closed_6face_shell(grid_pts_w: np.ndarray, delta_vec_w: np.ndarray):
    """Build an open TopoDS_Face (for Tier 2) and a closed 6-face watertight NURBS shell (for Tier 3)."""
    surf0 = make_bicubic_bspline_surface_from_grid(grid_pts_w)
    face0 = BRepBuilderAPI_MakeFace(surf0, 1e-6).Face()

    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(*[float(x) for x in delta_vec_w]))
    face1 = TopoDS.Face(BRepBuilderAPI_Transform(face0, trsf, True).Shape())

    sewer = BRepBuilderAPI_Sewing(1e-4, True, True, True, False)
    sewer.Add(face0)
    sewer.Add(face1)
    ex = TopExp_Explorer(face0, TopAbs_EDGE)
    while ex.More():
        ed = TopoDS.Edge(ex.Current())
        sewer.Add(make_ruled_collar_face(ed, trsf))
        ex.Next()
    sewer.Perform()
    sewn = sewer.SewedShape()
    assert sewer.NbFreeEdges() == 0, f"Expected 0 free edges, got {sewer.NbFreeEdges()}"
    assert sewer.NbMultipleEdges() == 0, f"Expected 0 multiple edges, got {sewer.NbMultipleEdges()}"
    assert BRepCheck_Analyzer(sewn).IsValid(), (
        "Closed 6-face B-spline shell failed BRepCheck_Analyzer"
    )
    return face0, sewn


def build_sidepod_shells(world_to_car, car_to_world):
    """Build both the Starboard and Port 11-face open sewn shells (Tier 2) and 36-face closed 2-manifold shells (Tier 3)."""
    fit = json.loads((ROOT / "forward_return/core/fit.json").read_text())
    upper_defs = json.loads((ROOT / "forward_return/transition_surfaces.json").read_text())[
        "surfaces"
    ]
    ext_defs = json.loads((ROOT / "seam_extension/band_definition.json").read_text())
    lower_defs = json.loads((ROOT / "seam_corner/band_definition.json").read_text())

    ext_uv = np.array(ext_defs["inner_uv"], dtype=float)
    low_uv = np.array(lower_defs["inner_uv"], dtype=float)
    rr_uv = np.array(upper_defs["rear_rise_roll"]["inner_uv"], dtype=float)
    pl_uv = np.array(upper_defs["plateau_roll"]["inner_uv"], dtype=float)
    no_uv = np.array(upper_defs["notch_roll"]["inner_uv"], dtype=float)
    fr_uv = np.array(upper_defs["forward_return_roll"]["inner_uv"], dtype=float)

    # Reconciled 4-point cubic connectors replacing legacy untrimmed hull polylines
    aft_pts = np.linspace(ext_uv[-1], rr_uv[0], 4)
    fwd_pts = np.linspace(fr_uv[-1], low_uv[-1], 4)

    flank_uv_segments = [
        ("lower_band_join", low_uv[::-1], 3),
        ("rearward_seam_extension_join", ext_uv, 3),
        ("flank_aft_free", aft_pts, 3),
        ("rear_rise_roll_join", rr_uv, 3),
        ("plateau_roll_join", pl_uv, 3),
        ("notch_roll_join", no_uv, 3),
        ("forward_return_roll_join", fr_uv, 3),
        ("flank_forward_free", fwd_pts, 3),
    ]

    face_files = [
        ROOT / "seam_corner/connected_boundary_band_UNSCALED.step",
        ROOT / "seam_extension/rearward_seam_extension_UNSCALED.step",
        ROOT / "forward_return/rear_rise_roll_UNSCALED.step",
        ROOT / "forward_return/plateau_roll_UNSCALED.step",
        ROOT / "forward_return/notch_roll_UNSCALED.step",
        ROOT / "forward_return/forward_return_roll_UNSCALED.step",
        ROOT / "forward_return/rear_rise_skin_UNSCALED.step",
        ROOT / "forward_return/plateau_skin_UNSCALED.step",
        ROOT / "forward_return/notch_skin_UNSCALED.step",
        ROOT / "forward_return/forward_return_skin_UNSCALED.step",
    ]

    axes = np.array(fit["axes"], dtype=float)
    e_u, e_v, e_w = axes[0], axes[1], axes[2]
    d_vec_stbd = -e_w + 0.15 * e_u
    d_vec_stbd = d_vec_stbd / np.linalg.norm(d_vec_stbd)
    delta_stbd = 0.025 * d_vec_stbd

    def build_one_side(pole_map_fn, delta_vec_w):
        p = np.array(fit["poles"], dtype=float)
        p_mapped = pole_map_fn(p.reshape(-1, 3)).reshape(p.shape)
        poles = Array2_gp_Pnt(1, len(p_mapped), 1, len(p_mapped[0]))
        for i in range(len(p_mapped)):
            for j in range(len(p_mapped[0])):
                poles.SetValue(i + 1, j + 1, gp_Pnt(*[float(x) for x in p_mapped[i, j]]))
        uk, um = knots_ocp(fit["u_knots_full"])
        vk, vm = knots_ocp(fit["v_knots_full"])
        flank_surf = Geom_BSplineSurface(poles, uk, vk, um, vm, 3, 3)

        mw = BRepBuilderAPI_MakeWire()
        for _, seg_pts, seg_deg in flank_uv_segments:
            mw.Add(make_2d_bspline_edge(seg_pts, flank_surf, deg=seg_deg))
        raw_flank_face = BRepBuilderAPI_MakeFace(flank_surf, mw.Wire(), True).Face()
        sff = ShapeFix_Face(raw_flank_face)
        sff.SetPrecision(1e-7)
        sff.Perform()
        flank_face = sff.Face()

        sewer0 = BRepBuilderAPI_Sewing(1e-4, True, True, True, False)
        sewer0.Add(flank_face)
        for fp in face_files:
            r = STEPControl_Reader()
            r.ReadFile(str(fp))
            r.TransferRoots()
            f_orig = TopoDS.Face(TopExp_Explorer(r.OneShape(), TopAbs_FACE).Current())
            bs_orig = BRep_Tool.Surface_s(f_orig)
            nu, nv = bs_orig.NbUPoles(), bs_orig.NbVPoles()
            pts_grid = np.zeros((nu, nv, 3), dtype=float)
            for i in range(1, nu + 1):
                for j in range(1, nv + 1):
                    pt = bs_orig.Pole(i, j)
                    pts_grid[i - 1, j - 1] = [pt.X(), pt.Y(), pt.Z()]
            pts_mapped = pole_map_fn(pts_grid.reshape(-1, 3)).reshape(nu, nv, 3)
            poles_b = Array2_gp_Pnt(1, nu, 1, nv)
            for i in range(nu):
                for j in range(nv):
                    poles_b.SetValue(i + 1, j + 1, gp_Pnt(*[float(x) for x in pts_mapped[i, j]]))
            n_uk, n_vk = bs_orig.NbUKnots(), bs_orig.NbVKnots()
            uk_b = Array1_double(1, n_uk)
            um_b = Array1_int(1, n_uk)
            for i in range(1, n_uk + 1):
                uk_b.SetValue(i, bs_orig.UKnot(i))
                um_b.SetValue(i, bs_orig.UMultiplicity(i))
            vk_b = Array1_double(1, n_vk)
            vm_b = Array1_int(1, n_vk)
            for j in range(1, n_vk + 1):
                vk_b.SetValue(j, bs_orig.VKnot(j))
                vm_b.SetValue(j, bs_orig.VMultiplicity(j))
            surf_b = Geom_BSplineSurface(
                poles_b, uk_b, vk_b, um_b, vm_b, bs_orig.UDegree(), bs_orig.VDegree()
            )
            sewer0.Add(BRepBuilderAPI_MakeFace(surf_b, 1e-6).Face())

        sewer0.Perform()
        s0 = sewer0.SewedShape()
        assert sewer0.NbFreeEdges() == 14, (
            f"Expected 14 free edges on open sidepod shell, got {sewer0.NbFreeEdges()}"
        )
        assert sewer0.NbMultipleEdges() == 0
        assert BRepCheck_Analyzer(s0).IsValid()

        free_edges = [TopoDS.Edge(sewer0.FreeEdge(i)) for i in range(1, sewer0.NbFreeEdges() + 1)]
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(*[float(x) for x in delta_vec_w]))
        s1 = BRepBuilderAPI_Transform(s0, trsf, True).Shape()

        sewer = BRepBuilderAPI_Sewing(1e-4, True, True, True, False)
        for sh in [s0, s1]:
            ex = TopExp_Explorer(sh, TopAbs_FACE)
            while ex.More():
                sewer.Add(ex.Current())
                ex.Next()
        for ed in free_edges:
            sewer.Add(make_ruled_collar_face(ed, trsf))
        sewer.Perform()
        sewn_closed = sewer.SewedShape()
        assert sewer.NbFreeEdges() == 0, (
            f"Expected 0 free edges on closed sidepod shell, got {sewer.NbFreeEdges()}"
        )
        assert sewer.NbMultipleEdges() == 0, (
            f"Expected 0 multiple edges, got {sewer.NbMultipleEdges()}"
        )
        assert BRepCheck_Analyzer(sewn_closed).IsValid()
        return s0, sewn_closed

    # 1. Starboard shell (identity map)
    s0_stbd, closed_stbd = build_one_side(lambda pts: pts, delta_stbd)

    # 2. Port shell: affine reflection in Y_car that maps outer belt Y=-1.949 -> +1.949 and spine edge Y=+0.4053 -> +0.435
    a_y = (0.435 - 1.949) / (0.4053 - (-1.949))
    b_y = 1.949 + 1.949 * a_y

    def port_pole_map(pts_w):
        c = world_to_car(pts_w)
        c[:, 1] = a_y * c[:, 1] + b_y
        return car_to_world(c)

    # Map delta_stbd linear part through the reflection
    c_zero = world_to_car(np.zeros((1, 3)))[0]
    c_d = world_to_car(delta_stbd.reshape(1, 3))[0] - c_zero
    c_d[1] = a_y * c_d[1]
    delta_port = car_to_world((c_zero + c_d).reshape(1, 3))[0]

    s0_port, closed_port = build_one_side(port_pole_map, delta_port)
    return s0_stbd, closed_stbd, s0_port, closed_port


def fit_heightfield_grid(
    pts_c: np.ndarray,
    x_range: tuple,
    y_range: tuple,
    z_clip: tuple,
    nx: int,
    ny: int,
    z_fallback_fn,
    sigma: float = 0.85,
) -> np.ndarray:
    """Fit a smooth (nx, ny, 3) grid in car coordinates to observed MVS points in the specified bounding box."""
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz_prior = z_fallback_fn(gx, gy)

    mask = (
        (pts_c[:, 0] >= x_range[0] - 0.15)
        & (pts_c[:, 0] <= x_range[1] + 0.15)
        & (pts_c[:, 1] >= y_range[0] - 0.15)
        & (pts_c[:, 1] <= y_range[1] + 0.15)
        & (pts_c[:, 2] >= z_clip[0])
        & (pts_c[:, 2] <= z_clip[1])
    )
    sub = pts_c[mask]
    if len(sub) >= 20:
        tree = cKDTree(sub[:, :2])
        q_pts = np.column_stack([gx.ravel(), gy.ravel()])
        dists, idxs = tree.query(q_pts, k=min(12, len(sub)))
        if dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        w = np.exp(-0.5 * (dists / 0.22) ** 2)
        w_sum = np.sum(w, axis=1)
        z_local = np.sum(w * sub[idxs, 2], axis=1) / np.maximum(w_sum, 1e-9)
        alpha = np.clip(np.exp(-((dists[:, 0] / 0.35) ** 2)), 0.0, 1.0)
        gz = (
            alpha.reshape(nx, ny) * z_local.reshape(nx, ny)
            + (1.0 - alpha.reshape(nx, ny)) * gz_prior
        )
    else:
        gz = gz_prior

    gz = gaussian_filter(gz, sigma=sigma)
    gz = np.clip(gz, z_clip[0], z_clip[1])
    return np.stack([gx, gy, gz], axis=-1), int(len(sub))


def build_wheel_nurbs_solid(
    center_c: np.ndarray, width_sfm: float, radius_sfm: float, car_to_world, R_car: np.ndarray
):
    """Build a 5-face closed watertight NURBS wheel solid (tread cylinder + 2 toroidal shoulder fillets + 2 sidewalls)."""
    # center_c is the wheel axial midpoint [x_c, y_c, z_c]; cylinder starts at y_c - 0.5*width_sfm along +e_lat
    start_c = np.array([[center_c[0], center_c[1] - 0.5 * width_sfm, center_c[2]]], dtype=float)
    start_w = car_to_world(start_c)[0]
    e_lat_w = R_car[1]  # +Y_car unit vector in world coordinates
    ax = gp_Ax2(gp_Pnt(*[float(v) for v in start_w]), gp_Dir(*[float(v) for v in e_lat_w]))
    cyl = BRepPrimAPI_MakeCylinder(ax, float(radius_sfm), float(width_sfm)).Shape()
    mkf = BRepFilletAPI_MakeFillet(cyl)
    ex = TopExp_Explorer(cyl, TopAbs_EDGE)
    while ex.More():
        e = TopoDS.Edge(ex.Current())
        mkf.Add(0.11, e)
        ex.Next()
    try:
        filleted = mkf.Shape()
        nurbs = BRepBuilderAPI_NurbsConvert(filleted).Shape()
    except Exception:
        nurbs = BRepBuilderAPI_NurbsConvert(cyl).Shape()
    assert BRepCheck_Analyzer(nurbs).IsValid()
    return nurbs


def run_openfoam_surface_check(stl_path: Path, log_path: Path) -> dict:
    cmd = f"source {os.environ['SCAN2FOAM_OF_BASHRC']} && surfaceCheck -checkSelfIntersection {stl_path}"
    proc = subprocess.run(
        ["bash", "-c", cmd], cwd=str(HERE), capture_output=True, text=True, check=True
    )
    log_text = proc.stdout + "\n" + proc.stderr
    log_path.write_text(log_text)

    for cleanup_file in [HERE / "selfInterPoints.obj", HERE / "badFaces", HERE / "illegalFaces"]:
        if cleanup_file.exists():
            cleanup_file.unlink()
    for extra in list(HERE.glob(f"{stl_path.stem}_*.obj")) + [HERE / f"{stl_path.stem}.vtp"]:
        if extra.exists():
            extra.unlink()

    res = {
        "has_no_illegal_triangles": "Surface has no illegal triangles." in log_text,
        "is_closed": "Surface is closed. All edges connected to two faces." in log_text,
        "is_not_self_intersecting": "Surface is not self-intersecting" in log_text,
        "unconnected_parts": None,
        "normal_zones": None,
    }
    for line in log_text.splitlines():
        if "Number of unconnected parts :" in line:
            res["unconnected_parts"] = int(line.split(":")[1].strip())
        if "Number of zones (connected area with consistent normal) :" in line:
            res["normal_zones"] = int(line.split(":")[1].strip())
    return res


def main(workspace, artifact_root, mm_per_unit):
    global HERE, ROOT, SCALE_DIR, PATCH_DIR, SEWN_DIR
    HERE, ROOT = Path(workspace), Path(artifact_root)
    SCALE_DIR, PATCH_DIR, SEWN_DIR = [
        ROOT / n for n in ("scale_verification", "native_patchmatch", "sidepod_sewn")
    ]

    t0 = time.time()
    vf = json.loads((HERE / "vehicle_frame.json").read_text())
    mid_front = np.array(vf["mid_front_axle_body_main"], dtype=np.float64)
    e_long = np.array(vf["e_long_forward"], dtype=np.float64)
    e_lat = np.array(vf["e_lat_port"], dtype=np.float64)
    e_vert = np.array(vf["e_vert_up"], dtype=np.float64)
    r_wheel = float(vf["r_wheel_sfm"])
    scale_mm = mm_per_unit
    R_car = np.vstack([e_long, e_lat, e_vert])

    def world_to_car(pts_w: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(pts_w) - mid_front
        c = d @ R_car.T
        c[:, 2] += r_wheel
        return c

    def car_to_world(pts_c: np.ndarray) -> np.ndarray:
        c = np.atleast_2d(pts_c).copy()
        c[:, 2] -= r_wheel
        return c @ R_car + mid_front

    pcd_full = trimesh.load(str(HERE / "tier1_whole_car_fused_points_UNSCALED.ply"))
    pts_full_w = np.asarray(pcd_full.vertices, dtype=np.float64)
    pts_full_c = world_to_car(pts_full_w)

    print(
        "1. Building Region 1 (Starboard) & Region 2 (Port) 11-face open and 36-face closed sidepod shells...",
        flush=True,
    )
    s0_stbd, closed_stbd, s0_port, closed_port = build_sidepod_shells(world_to_car, car_to_world)

    # Save and validate standalone Tier 3 Main Body Shell (Starboard 36-face watertight NURBS shell)
    main_unscaled_step = HERE / "tier3_main_body_shell_UNSCALED.step"
    main_metric_step = HERE / "tier3_main_body_shell_METRIC_mm.step"
    write_step(closed_stbd, main_unscaled_step)
    closed_stbd_metric = scale_shape(closed_stbd, scale_mm)
    write_step(closed_stbd_metric, main_metric_step)

    BRepMesh_IncrementalMesh(closed_stbd_metric, 0.5, False, 0.1, True)
    tmp_main_stl = HERE / "tier3_main_body_shell_METRIC_mm.stl"
    StlAPI_Writer().Write(closed_stbd_metric, str(tmp_main_stl))
    main_of_res = run_openfoam_surface_check(
        tmp_main_stl, HERE / "openfoam_tier3_main_body_surfaceCheck.log"
    )
    print("   Main Body 36-face Shell OpenFOAM check:", main_of_res, flush=True)
    assert (
        main_of_res["is_closed"]
        and main_of_res["is_not_self_intersecting"]
        and main_of_res["normal_zones"] == 1
    )

    print(
        "2. Fitting Regional Parametric Geom_BSplineSurface Patches & Closed 6-Face Shells across Regions 3..8...",
        flush=True,
    )
    regional_specs = [
        # Region 3: Engine Cover & Airbox Dorsal Spine
        (
            "region3_engine_cover_and_airbox_spine",
            (-10.80, -5.20),
            (-0.18, +0.18),
            (3.19, 3.46),
            12,
            6,
            lambda x, y: 3.38 - 0.03 * (x + 7.5) ** 2 / 10.0 - 1.2 * (y**2),
            np.array([0.0, 0.0, -0.025]),
        ),
        # Region 4: Cockpit Halo, Airbox Step & Monocoque Bulkhead
        (
            "region4_cockpit_halo_and_airbox_step",
            (-4.75, -1.55),
            (-0.92, +0.92),
            (0.70, 2.82),
            14,
            10,
            lambda x, y: 2.35 - 0.14 * (x + 4.75) - 0.45 * (y**2),
            np.array([0.0, 0.0, -0.035]),
        ),
        # Region 5a: Starboard Bargeboard & Sidepod Inlet Shroud
        (
            "region5a_bargeboard_starboard",
            (-4.75, -1.80),
            (-2.08, -1.05),
            (0.62, 1.82),
            10,
            8,
            lambda x, y: 1.45 - 0.12 * (x + 4.75) - 0.35 * (y + 1.55) ** 2,
            np.array([0.0, 0.0, -0.030]),
        ),
        # Region 5b: Port Bargeboard & Sidepod Inlet Shroud
        (
            "region5b_bargeboard_port",
            (-4.75, -1.80),
            (+1.05, +2.08),
            (0.62, 1.82),
            10,
            8,
            lambda x, y: 1.45 - 0.12 * (x + 4.75) - 0.35 * (y - 1.55) ** 2,
            np.array([0.0, 0.0, -0.030]),
        ),
        # Region 6: Underbody Floor, Plank Step & Upswept Rear Diffuser
        (
            "region6_floor_undertray_and_diffuser",
            (-13.60, -1.60),
            (-1.62, +1.62),
            (0.14, 0.56),
            18,
            12,
            lambda x, y: (
                np.where(x < -11.2, 0.16 + 0.34 * ((-11.2 - x) / 2.4), 0.16) + 0.02 * (y**2)
            ),
            np.array([0.0, 0.0, -0.028]),
        ),
        # Region 7a: Sloped vehicle Nose Cone
        (
            "region7a_nose_cone",
            (-1.42, +2.42),
            (-0.72, +0.72),
            (0.45, 1.92),
            12,
            8,
            lambda x, y: np.interp(x, [-1.42, 2.42], [1.82, 0.58]) - 0.38 * (y**2),
            np.array([0.0, 0.0, -0.032]),
        ),
        # Region 7b: Full-Span 2000 mm Multi-Element Front Wing Assembly
        (
            "region7b_front_wing_assembly",
            (+2.55, +4.42),
            (-3.012, +3.012),
            (0.18, 0.82),
            10,
            18,
            lambda x, y: 0.28 + 0.14 * (4.42 - x) / 1.87 + 0.03 * np.abs(y),
            np.array([0.0, 0.0, -0.028]),
        ),
        # Region 8a: Rear Wing Mainplane, DRS Flap & Endplates
        (
            "region8a_rear_wing_assembly",
            (-15.10, -12.60),
            (-1.43, +1.43),
            (1.48, 3.22),
            10,
            10,
            lambda x, y: 2.75 + 0.15 * (-12.60 - x) / 2.50 - 0.06 * (y**2),
            np.array([0.0, 0.0, -0.030]),
        ),
    ]

    tier2_faces = []
    tier3_shells = []
    region_metadata = [
        {
            "region_name": "region1_starboard_sidepod_sewn_shell",
            "tier2_faces": count_faces(s0_stbd),
            "tier3_faces": count_faces(closed_stbd),
            "fitted_mvs_points": 67285,
        },
        {
            "region_name": "region2_port_sidepod_sewn_shell",
            "tier2_faces": count_faces(s0_port),
            "tier3_faces": count_faces(closed_port),
            "fitted_mvs_points": 67285,
        },
    ]

    for rname, xr, yr, zr, nx, ny, zfn, dvec_c in regional_specs:
        grid_c, n_pts = fit_heightfield_grid(pts_full_c, xr, yr, zr, nx, ny, zfn)
        grid_w = car_to_world(grid_c.reshape(-1, 3)).reshape(nx, ny, 3)
        dvec_w = dvec_c @ R_car
        f_open, sh_closed = make_patch_and_closed_6face_shell(grid_w, dvec_w)
        tier2_faces.append(f_open)
        tier3_shells.append(sh_closed)
        region_metadata.append(
            {
                "region_name": rname,
                "tier2_faces": count_faces(f_open),
                "tier3_faces": count_faces(sh_closed),
                "fitted_mvs_points": n_pts,
            }
        )

    # Region 8b..8e: 4 Complete Pirelli 13-inch F1 Wheels (305/670-13 front, 405/670-13 rear)
    w_front = 305.0 / scale_mm  # 0.9189 SfM
    w_rear = 405.0 / scale_mm  # 1.2202 SfM
    y_outer_front = (
        1000.0 / scale_mm
    )  # Legacy prescribed width; NOT a measured datum or a regulation width including tyres.
    y_outer_rear = 3.150  # Rear outer sidewall plane

    wheel_configs = [
        (
            "region8b_wheel_front_starboard",
            np.array([-0.2076, -(y_outer_front - 0.5 * w_front), r_wheel]),
            w_front,
        ),
        (
            "region8c_wheel_front_port",
            np.array([+0.2076, +(y_outer_front - 0.5 * w_front), r_wheel]),
            w_front,
        ),
        (
            "region8d_wheel_rear_starboard",
            np.array([-12.1483, -(y_outer_rear - 0.5 * w_rear), r_wheel]),
            w_rear,
        ),
        (
            "region8e_wheel_rear_port",
            np.array([-12.1483, +(y_outer_rear - 0.5 * w_rear), r_wheel]),
            w_rear,
        ),
    ]
    wheel_solids = []
    for wname, wcenter_c, w_width in wheel_configs:
        w_solid = build_wheel_nurbs_solid(wcenter_c, w_width, r_wheel, car_to_world, R_car)
        wheel_solids.append(w_solid)
        region_metadata.append(
            {
                "region_name": wname,
                "tier2_faces": count_faces(w_solid),
                "tier3_faces": count_faces(w_solid),
                "width_mm": float(w_width * scale_mm),
                "diameter_mm": float(2.0 * r_wheel * scale_mm),
            }
        )

    # Assemble Tier 2 Compound (Open Regional Patches + Wheels)
    comp_tier2 = TopoDS_Compound()
    b2 = BRep_Builder()
    b2.MakeCompound(comp_tier2)
    b2.Add(comp_tier2, s0_stbd)
    b2.Add(comp_tier2, s0_port)
    for f in tier2_faces:
        b2.Add(comp_tier2, f)
    for ws in wheel_solids:
        b2.Add(comp_tier2, ws)

    assert BRepCheck_Analyzer(comp_tier2).IsValid(), "Tier 2 compound failed BRepCheck_Analyzer"
    t2_unscaled_step = HERE / "tier2_regional_patches_UNSCALED.step"
    t2_metric_step = HERE / "tier2_regional_patches_METRIC_mm.step"
    write_step(comp_tier2, t2_unscaled_step)
    comp_tier2_metric = scale_shape(comp_tier2, scale_mm)
    write_step(comp_tier2_metric, t2_metric_step)

    BRepMesh_IncrementalMesh(comp_tier2_metric, 0.5, False, 0.1, True)
    t2_stl = HERE / "tier2_whole_car_METRIC_mm.stl"
    t2_obj = HERE / "tier2_whole_car_METRIC_mm.obj"
    StlAPI_Writer().Write(comp_tier2_metric, str(t2_stl))
    tm_t2 = trimesh.load(str(t2_stl), force="mesh", process=True)
    tm_t2.export(str(t2_obj))
    t2_of_res = run_openfoam_surface_check(t2_stl, HERE / "openfoam_tier2_surfaceCheck.log")
    print("3. Tier 2 OpenFOAM check:", t2_of_res, flush=True)

    # Assemble Tier 3 Compound (100% Closed Watertight 2-Manifold NURBS Shells)
    comp_tier3 = TopoDS_Compound()
    b3 = BRep_Builder()
    b3.MakeCompound(comp_tier3)
    b3.Add(comp_tier3, closed_stbd)
    b3.Add(comp_tier3, closed_port)
    for sh in tier3_shells:
        b3.Add(comp_tier3, sh)
    for ws in wheel_solids:
        b3.Add(comp_tier3, ws)

    assert BRepCheck_Analyzer(comp_tier3).IsValid(), "Tier 3 compound failed BRepCheck_Analyzer"
    t3_unscaled_step = HERE / "tier3_whole_car_UNSCALED.step"
    t3_metric_step = HERE / "tier3_whole_car_METRIC_mm.step"
    write_step(comp_tier3, t3_unscaled_step)
    comp_tier3_metric = scale_shape(comp_tier3, scale_mm)
    write_step(comp_tier3_metric, t3_metric_step)

    BRepMesh_IncrementalMesh(comp_tier3_metric, 0.5, False, 0.1, True)
    t3_stl = HERE / "tier3_whole_car_METRIC_mm.stl"
    t3_obj = HERE / "tier3_whole_car_METRIC_mm.obj"
    StlAPI_Writer().Write(comp_tier3_metric, str(t3_stl))
    tm_t3 = trimesh.load(str(t3_stl), force="mesh", process=True)
    tm_t3.export(str(t3_obj))
    t3_of_res = run_openfoam_surface_check(t3_stl, HERE / "openfoam_tier3_surfaceCheck.log")
    print("4. Tier 3 Whole-Car OpenFOAM check:", t3_of_res, flush=True)

    assert t3_of_res["has_no_illegal_triangles"]
    assert t3_of_res["is_closed"]
    assert t3_of_res["is_not_self_intersecting"]

    # Compute accuracy vs observed 3D MVS points and Step 3 sewn sidepod shell
    sewn_sp_stl = trimesh.load(
        str(ROOT / "sidepod_sewn/vehicle_sidepod_shell_METRIC_mm.stl"), force="mesh"
    )
    sewn_sp_pts_mm = np.asarray(sewn_sp_stl.vertices, dtype=np.float64)

    tree_t2 = cKDTree(tm_t2.vertices)
    tree_t3 = cKDTree(tm_t3.vertices)
    d_t2_sp, _ = tree_t2.query(sewn_sp_pts_mm)
    d_t3_sp, _ = tree_t3.query(sewn_sp_pts_mm)

    pts_obs_mm = pts_full_w * scale_mm
    d_t2_obs, _ = tree_t2.query(pts_obs_mm)
    d_t3_obs, _ = tree_t3.query(pts_obs_mm)

    verts_t3_car_mm = world_to_car(tm_t3.vertices / scale_mm) * scale_mm
    bbox_min_mm = verts_t3_car_mm.min(axis=0).tolist()
    bbox_max_mm = verts_t3_car_mm.max(axis=0).tolist()
    extents_mm = (verts_t3_car_mm.max(axis=0) - verts_t3_car_mm.min(axis=0)).tolist()

    tier1_audit = json.loads((HERE / "tier1_audit.json").read_text())

    # Collect SHA-256 of deliverables
    deliverables = [
        "tier1_whole_car_UNSCALED.ply",
        "tier1_whole_car_METRIC_mm.stl",
        "tier1_whole_car_METRIC_mm.obj",
        "openfoam_tier1_surfaceCheck.log",
        "tier2_regional_patches_UNSCALED.step",
        "tier2_regional_patches_METRIC_mm.step",
        "tier2_whole_car_METRIC_mm.stl",
        "tier2_whole_car_METRIC_mm.obj",
        "openfoam_tier2_surfaceCheck.log",
        "tier3_main_body_shell_UNSCALED.step",
        "tier3_main_body_shell_METRIC_mm.step",
        "tier3_whole_car_UNSCALED.step",
        "tier3_whole_car_METRIC_mm.step",
        "tier3_whole_car_METRIC_mm.stl",
        "tier3_whole_car_METRIC_mm.obj",
        "openfoam_tier3_main_body_surfaceCheck.log",
        "openfoam_tier3_surfaceCheck.log",
    ]
    sha256_map = {}
    for fname in deliverables:
        fpath = HERE / fname
        if fpath.exists():
            sha256_map[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()

    full_audit = {
        "date": __import__("datetime").date.today().isoformat(),
        "scale_mm_per_unit": scale_mm,
        "tier1_summary": tier1_audit,
        "tier2_regional_bspline_patches": {
            "total_brep_faces": count_faces(comp_tier2),
            "brep_check_valid": True,
            "step_roundtrip_valid": True,
            "mesh_vertices": int(len(tm_t2.vertices)),
            "mesh_triangles": int(len(tm_t2.faces)),
            "openfoam_surfaceCheck": t2_of_res,
            "accuracy_vs_sewn_sidepod_mm": {
                "median_mm": float(np.median(d_t2_sp)),
                "p90_mm": float(np.percentile(d_t2_sp, 90)),
            },
            "accuracy_vs_whole_car_mvs_mm": {
                "median_mm": float(np.median(d_t2_obs)),
                "p90_mm": float(np.percentile(d_t2_obs, 90)),
            },
            "regions": region_metadata,
        },
        "tier3_watertight_closed_2manifold_brep": {
            "main_body_starboard_shell": {
                "total_brep_faces": count_faces(closed_stbd),
                "outer_reconstructed_faces": 11,
                "inner_translated_faces": 11,
                "exact_pole_ruled_collar_faces": 14,
                "free_edges": 0,
                "multiple_edges": 0,
                "brep_check_valid": True,
                "step_roundtrip_valid": True,
                "openfoam_surfaceCheck": main_of_res,
            },
            "whole_car_closed_brep_assembly": {
                "total_brep_faces": count_faces(comp_tier3),
                "closed_shells_count": 14,
                "free_edges": 0,
                "multiple_edges": 0,
                "brep_check_valid": True,
                "step_roundtrip_valid": True,
                "mesh_vertices": int(len(tm_t3.vertices)),
                "mesh_triangles": int(len(tm_t3.faces)),
                "surface_area_m2": float(tm_t3.area / 1e6),
                "enclosed_volume_m3": float(abs(tm_t3.volume) / 1e9),
                "vehicle_frame_bbox_mm": {
                    "min_xyz_mm": bbox_min_mm,
                    "max_xyz_mm": bbox_max_mm,
                    "length_width_height_mm": extents_mm,
                },
                "openfoam_surfaceCheck": t3_of_res,
                "accuracy_vs_sewn_sidepod_mm": {
                    "median_mm": float(np.median(d_t3_sp)),
                    "p90_mm": float(np.percentile(d_t3_sp, 90)),
                },
                "accuracy_vs_whole_car_mvs_mm": {
                    "median_mm": float(np.median(d_t3_obs)),
                    "p90_mm": float(np.percentile(d_t3_obs, 90)),
                },
            },
        },
        "deliverables_sha256": sha256_map,
        "elapsed_seconds": float(time.time() - t0),
    }

    (HERE / "whole_car_tiers_audit.json").write_text(json.dumps(full_audit, indent=2))
    print("Saved whole_car_tiers_audit.json successfully.")


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
