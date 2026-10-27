"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Transform,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.collections import Array1_double, Array1_gp_Pnt2d, Array1_int, Array2_gp_Pnt
from OCP.Geom import Geom_BSplineSurface
from OCP.Geom2d import Geom2d_BSplineCurve
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shell
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound
from scipy.interpolate import make_interp_spline


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


def read_step_shape(path):
    r = STEPControl_Reader()
    assert r.ReadFile(str(path)) == IFSelect_RetDone, f"Failed to read {path}"
    r.TransferRoots()
    return r.OneShape()


def write_step_checked(shape, path):
    assert BRepCheck_Analyzer(shape).IsValid(), f"Invalid shape before STEP write: {path}"
    w = STEPControl_Writer()
    assert w.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert w.Write(str(path)) == IFSelect_RetDone
    rt = read_step_shape(path)
    assert BRepCheck_Analyzer(rt).IsValid(), f"Invalid shape after STEP round-trip: {path}"
    return rt


def sample_edge_3d(edge, n=101):
    first, last = BRep_Tool.Range_s(edge)
    c3d = BRep_Tool.Curve_s(edge, first, last)
    ts = np.linspace(first, last, n)
    pts = []
    for t in ts:
        p = c3d.Value(float(t))
        pts.append([p.X(), p.Y(), p.Z()])
    pts = np.array(pts, dtype=float)
    if edge.Orientation() == TopAbs_REVERSED:
        pts = pts[::-1]
    return pts


def classify_free_edge(pts, upper_defs, lower_defs, ext_defs):
    """Assign a human-readable physical boundary name and neighboring car region to each exterior free edge."""
    mid = pts[len(pts) // 2]
    p0, p1 = pts[0], pts[-1]
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    # Check proximity to known outer curves of the 11-face assembly
    best_name = None
    best_neighbor = None
    best_dist = 1e9

    # 1. Upper skin outer edges (4 segments)
    for seg in ["rear_rise", "plateau", "notch", "forward_return"]:
        poles = np.array(upper_defs[f"{seg}_skin"]["poles"])
        outer_pts = poles[:, -1, :]
        d = min(
            np.linalg.norm(outer_pts - mid, axis=1).min(),
            0.5
            * (
                np.linalg.norm(outer_pts - p0, axis=1).min()
                + np.linalg.norm(outer_pts - p1, axis=1).min()
            ),
        )
        if d < best_dist:
            best_dist = d
            best_name = f"upper_skin_outer_{seg}"
            best_neighbor = "engine_cover_and_airbox_spine"

    # 2. Upper rear_rise aft transverse end (roll + skin at u=0)
    for kind in ["roll", "skin"]:
        poles = np.array(upper_defs[f"rear_rise_{kind}"]["poles"])
        aft_pts = poles[0, :, :]
        d = 0.5 * (
            np.linalg.norm(aft_pts - p0, axis=1).min() + np.linalg.norm(aft_pts - p1, axis=1).min()
        )
        if d < best_dist:
            best_dist = d
            best_name = f"upper_rear_rise_{kind}_aft_end"
            best_neighbor = "rear_cooling_exit_and_cola"

    # 3. Upper forward_return forward transverse end (roll + skin at u=1)
    for kind in ["roll", "skin"]:
        poles = np.array(upper_defs[f"forward_return_{kind}"]["poles"])
        fwd_pts = poles[-1, :, :]
        d = 0.5 * (
            np.linalg.norm(fwd_pts - p0, axis=1).min() + np.linalg.norm(fwd_pts - p1, axis=1).min()
        )
        if d < best_dist:
            best_dist = d
            best_name = f"upper_forward_return_{kind}_fwd_end"
            best_neighbor = "cockpit_halo_and_airbox_step"

    # 4. Lower band outer edge, forward-inlet end, and aft step
    lower_poles = np.array(lower_defs["poles"])
    candidates = [
        ("lower_band_outer_edge", "floor_undertray_and_inlet_lip", lower_poles[:, -1, :]),
        (
            "lower_band_inlet_top_end",
            "sidepod_inlet_interior_and_bargeboard",
            lower_poles[-1, :, :],
        ),
        ("lower_band_aft_end", "floor_undertray_mid_step", lower_poles[0, :, :]),
    ]
    ext_poles = np.array(ext_defs["poles"])
    candidates.extend(
        [
            (
                "rearward_seam_extension_outer_edge",
                "floor_undertray_rear_diffuser_ramp",
                ext_poles[:, -1, :],
            ),
            (
                "rearward_seam_extension_aft_end",
                "rear_cooling_exit_and_floor_kick",
                ext_poles[0, :, :],
            ),
            ("rearward_seam_extension_fwd_end", "floor_undertray_mid_step", ext_poles[-1, :, :]),
        ]
    )
    for cname, cneigh, cpts in candidates:
        d = 0.5 * (
            np.linalg.norm(cpts - p0, axis=1).min() + np.linalg.norm(cpts - p1, axis=1).min()
        )
        if d < best_dist:
            best_dist = d
            best_name = cname
            best_neighbor = cneigh

    # 5. Check aft and forward vertical flank free edges
    if best_dist > 1e-3:
        if mid[0] < -0.5:
            best_name = "flank_aft_free_edge"
            best_neighbor = "rear_cooling_exit_and_cola"
        else:
            best_name = "flank_forward_inlet_free_edge"
            best_neighbor = "bargeboard_and_sidepod_inlet_shroud"

    return {
        "name": best_name,
        "neighbor_region": best_neighbor,
        "length_unscaled": length,
        "start_xyz_unscaled": p0.tolist(),
        "end_xyz_unscaled": p1.tolist(),
        "midpoint_xyz_unscaled": mid.tolist(),
        "match_distance_unscaled": float(best_dist),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--upper-dir",
        type=Path,
        required=True,
        help="Directory containing transition_surfaces.json, core/fit.json, and 8 upper STEP faces",
    )
    ap.add_argument(
        "--scale-json",
        type=Path,
        required=True,
        help="Path to scale_audit.json for metric scaling",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for sewn shell artefacts",
    )
    ap.add_argument("--artifact-root", type=Path, required=True)
    args = ap.parse_args()

    R = args.out_dir
    R.mkdir(parents=True, exist_ok=False)
    (R / "inspection").mkdir(exist_ok=True)

    vehicle_root = args.artifact_root
    upper_dir = args.upper_dir
    seam_ext_dir = vehicle_root / "seam_extension"
    seam_corner_dir = vehicle_root / "seam_corner"

    # Load JSON definitions
    fit = json.loads((upper_dir / "core/fit.json").read_text())
    upper_json = json.loads((upper_dir / "transition_surfaces.json").read_text())
    upper_defs = upper_json["surfaces"]
    ext_defs = json.loads((seam_ext_dir / "band_definition.json").read_text())
    lower_defs = json.loads((seam_corner_dir / "band_definition.json").read_text())

    # Metric export is explicit; missing scale evidence must not fall back to a historical estimate.
    rms = json.loads(args.scale_json.read_text())["recommended_metric_scale"]
    scale_mm_per_unit = float(rms["primary_mm_per_unit"])
    scale_show_670_mm_per_unit = float(rms["sensitivity_upper_mm_per_unit"])
    scale_midpoint_mm_per_unit = float(rms["midpoint_mm_per_unit"])
    scale_statistical_uncertainty_pct = float(rms["relative_statistical_uncertainty_pct"])
    scale_uncertainty_pct = float(rms["relative_sensitivity_pct"])
    scale_source = str(args.scale_json)
    if scale_mm_per_unit <= 0:
        raise ValueError("Scale must be positive and independently justified")

    # 1. Build untrimmed flank Geom_BSplineSurface
    p = np.array(fit["poles"], dtype=float)
    poles = Array2_gp_Pnt(1, len(p), 1, len(p[0]))
    for i in range(len(p)):
        for j in range(len(p[0])):
            poles.SetValue(i + 1, j + 1, gp_Pnt(*p[i, j]))
    uk, um = knots_ocp(fit["u_knots_full"])
    vk, vm = knots_ocp(fit["v_knots_full"])
    deg_u = int(fit.get("degree_u", fit.get("degree", 3)))
    deg_v = int(fit.get("degree_v", fit.get("degree", 3)))
    flank_surf = Geom_BSplineSurface(poles, uk, vk, um, vm, deg_u, deg_v)

    # 2. Audit shared UV boundary curves on flank trim_uv
    trim_uv = np.array(fit["trim_uv"], dtype=float)
    ext_uv = np.array(ext_defs["inner_uv"], dtype=float)
    low_uv = np.array(lower_defs["inner_uv"], dtype=float)
    rr_uv = np.array(upper_defs["rear_rise_roll"]["inner_uv"], dtype=float)
    pl_uv = np.array(upper_defs["plateau_roll"]["inner_uv"], dtype=float)
    no_uv = np.array(upper_defs["notch_roll"]["inner_uv"], dtype=float)
    fr_uv = np.array(upper_defs["forward_return_roll"]["inner_uv"], dtype=float)

    def find_trim_idx(pt):
        d = np.linalg.norm(trim_uv - pt, axis=1)
        idx = int(np.argmin(d))
        assert d[idx] < 1e-6, f"Point {pt} not on trim_uv (min dist {d[idx]})"
        return idx

    i_ext_0, i_ext_1 = find_trim_idx(ext_uv[0]), find_trim_idx(ext_uv[-1])
    i_low_0, i_low_1 = find_trim_idx(low_uv[0]), find_trim_idx(low_uv[-1])
    i_fr_1, i_rr_0 = find_trim_idx(fr_uv[-1]), find_trim_idx(rr_uv[0])

    # Extract the two free-boundary UV segments on the flank (aft connector and forward connector)
    # Loop order around trim_uv: low[::-1] (2->402) -> ext (402->802) -> aft_pts (802->849) ->
    # rr (849->1099) -> pl (1099->1349) -> no (1349->1599) -> fr (1599->1849) -> fwd_pts (1849->2)
    aft_pts = trim_uv[i_ext_1 : i_rr_0 + 1]
    fwd_pts = np.vstack([trim_uv[i_fr_1:], trim_uv[: i_low_1 + 1]])

    # Build the 8 smooth 2D BSpline edges of the reconciled flank wire
    flank_uv_segments = [
        ("lower_band_join", low_uv[::-1], 3),
        ("rearward_seam_extension_join", ext_uv, 3),
        ("flank_aft_free", aft_pts, 1),
        ("rear_rise_roll_join", rr_uv, 3),
        ("plateau_roll_join", pl_uv, 3),
        ("notch_roll_join", no_uv, 3),
        ("forward_return_roll_join", fr_uv, 3),
        ("flank_forward_free", fwd_pts, 1),
    ]

    mw = BRepBuilderAPI_MakeWire()
    for seg_name, seg_pts, seg_deg in flank_uv_segments:
        mw.Add(make_2d_bspline_edge(seg_pts, flank_surf, deg=seg_deg))
    assert mw.IsDone(), "Failed to build reconciled flank wire"

    raw_flank_face = BRepBuilderAPI_MakeFace(flank_surf, mw.Wire(), True).Face()
    sff = ShapeFix_Face(raw_flank_face)
    sff.SetPrecision(1e-7)
    sff.Perform()
    flank_face = sff.Face()
    assert BRepCheck_Analyzer(flank_face).IsValid(), "Reconciled flank face is not BRepCheck valid"

    # Save the reconciled flank face STEP
    write_step_checked(flank_face, R / "sidepod_flank_reconciled_UNSCALED.step")

    # 3. Load the 10 adjacent band/skin STEP faces
    face_files = {
        "flank_interior": R / "sidepod_flank_reconciled_UNSCALED.step",
        "connected_boundary_band": seam_corner_dir / "connected_boundary_band_UNSCALED.step",
        "rearward_seam_extension": seam_ext_dir / "rearward_seam_extension_UNSCALED.step",
        "rear_rise_roll": upper_dir / "rear_rise_roll_UNSCALED.step",
        "plateau_roll": upper_dir / "plateau_roll_UNSCALED.step",
        "notch_roll": upper_dir / "notch_roll_UNSCALED.step",
        "forward_return_roll": upper_dir / "forward_return_roll_UNSCALED.step",
        "rear_rise_skin": upper_dir / "rear_rise_skin_UNSCALED.step",
        "plateau_skin": upper_dir / "plateau_skin_UNSCALED.step",
        "notch_skin": upper_dir / "notch_skin_UNSCALED.step",
        "forward_return_skin": upper_dir / "forward_return_skin_UNSCALED.step",
    }

    faces = {}
    for name, fpath in face_files.items():
        sh = read_step_shape(fpath)
        exp = TopExp_Explorer(sh, TopAbs_FACE)
        assert exp.More(), f"No face found in {fpath}"
        faces[name] = TopoDS.Face(exp.Current())

    # 4. Sew all 11 faces using BRepBuilderAPI_Sewing
    sewing_tol = 1e-4
    sewer = BRepBuilderAPI_Sewing(sewing_tol, True, True, True, False)
    for name, f in faces.items():
        sewer.Add(f)
    sewer.Perform()
    sewn_shape = sewer.SewedShape()

    # Check shell structure and orient consistently
    shell_exp = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
    shells = []
    while shell_exp.More():
        shells.append(TopoDS.Shell(shell_exp.Current()))
        shell_exp.Next()
    assert len(shells) == 1, f"Expected 1 TopoDS_Shell, got {len(shells)}"

    sfs = ShapeFix_Shell(shells[0])
    sfs.FixFaceOrientation(shells[0], True, False)
    sewn_shell = sfs.Shell()
    assert BRepCheck_Analyzer(sewn_shell).IsValid(), "Sewn TopoDS_Shell failed BRepCheck_Analyzer"

    # 5. Detailed topological edge valence audit on sewn_shell + sewer diagnostics
    assert sewer.NbFreeEdges() == 14, f"Sewer NbFreeEdges={sewer.NbFreeEdges()}"
    assert sewer.NbContigousEdges() == 17, f"Sewer NbContigousEdges={sewer.NbContigousEdges()}"
    assert sewer.NbMultipleEdges() == 0, f"Sewer NbMultipleEdges={sewer.NbMultipleEdges()}"
    assert sewer.NbDegeneratedShapes() == 0, (
        f"Sewer NbDegeneratedShapes={sewer.NbDegeneratedShapes()}"
    )

    free_edges = [TopoDS.Edge(sewer.FreeEdge(i)) for i in range(1, sewer.NbFreeEdges() + 1)]
    contiguous_edges = [
        TopoDS.Edge(sewer.ContigousEdge(i)) for i in range(1, sewer.NbContigousEdges() + 1)
    ]
    multiple_edges = [
        TopoDS.Edge(sewer.MultipleEdge(i)) for i in range(1, sewer.NbMultipleEdges() + 1)
    ]
    degenerated_edges = []
    eexp = TopExp_Explorer(sewn_shell, TopAbs_EDGE)
    while eexp.More():
        ed = TopoDS.Edge(eexp.Current())
        if BRep_Tool.Degenerated_s(ed):
            degenerated_edges.append(ed)
        eexp.Next()

    assert len(multiple_edges) == 0, f"Found {len(multiple_edges)} non-manifold edges!"
    assert len(degenerated_edges) == 0, f"Found {len(degenerated_edges)} degenerated edges!"
    assert len(contiguous_edges) == 17, (
        f"Expected 17 internal contiguous edges, got {len(contiguous_edges)}"
    )
    assert len(free_edges) == 14, f"Expected 14 exterior free edges, got {len(free_edges)}"

    # Verify STEP round-trip of the unscaled sewn shell
    unscaled_step_path = R / "vehicle_sidepod_shell_UNSCALED.step"
    rt_shape = write_step_checked(sewn_shell, unscaled_step_path)
    rt_shells = 0
    rt_exp = TopExp_Explorer(rt_shape, TopAbs_SHELL)
    while rt_exp.More():
        rt_shells += 1
        rt_exp.Next()
    rt_faces = 0
    rt_fexp = TopExp_Explorer(rt_shape, TopAbs_FACE)
    while rt_fexp.More():
        rt_faces += 1
        rt_fexp.Next()
    assert rt_shells == 1 and rt_faces == 11, (
        f"Round-trip shell count={rt_shells}, face count={rt_faces}"
    )

    # 6. Export named exterior free-boundary curves (UNSCALED and METRIC_mm)
    builder = BRep_Builder()
    free_compound_unscaled = TopoDS_Compound()
    builder.MakeCompound(free_compound_unscaled)
    free_compound_metric = TopoDS_Compound()
    builder.MakeCompound(free_compound_metric)

    trsf_metric = gp_Trsf()
    trsf_metric.SetScale(gp_Pnt(0.0, 0.0, 0.0), float(scale_mm_per_unit))

    free_edge_records = []
    used_names = {}
    for idx, edge in enumerate(free_edges):
        pts = sample_edge_3d(edge, n=101)
        rec = classify_free_edge(pts, upper_defs, lower_defs, ext_defs)
        base_name = rec["name"]
        if base_name in used_names:
            used_names[base_name] += 1
            rec["name"] = f"{base_name}_{used_names[base_name]}"
        else:
            used_names[base_name] = 1
        rec["edge_index"] = idx + 1
        rec["length_metric_mm"] = float(rec["length_unscaled"] * scale_mm_per_unit)
        rec["start_xyz_metric_mm"] = (
            np.array(rec["start_xyz_unscaled"]) * scale_mm_per_unit
        ).tolist()
        rec["end_xyz_metric_mm"] = (np.array(rec["end_xyz_unscaled"]) * scale_mm_per_unit).tolist()
        free_edge_records.append(rec)

        builder.Add(free_compound_unscaled, edge)
        edge_metric = TopoDS.Edge(BRepBuilderAPI_Transform(edge, trsf_metric, True).Shape())
        builder.Add(free_compound_metric, edge_metric)

    write_step_checked(free_compound_unscaled, R / "free_boundaries_UNSCALED.step")
    write_step_checked(free_compound_metric, R / "free_boundaries_METRIC_mm.step")

    # 7. Export Metric-Scaled Sewn Shell STEP (mm)
    metric_shell_shape = BRepBuilderAPI_Transform(sewn_shell, trsf_metric, True).Shape()
    metric_step_path = R / "vehicle_sidepod_shell_METRIC_mm.step"
    write_step_checked(metric_shell_shape, metric_step_path)

    # Surface area props
    props_unscaled = GProp_GProps()
    BRepGProp.SurfaceProperties_s(sewn_shell, props_unscaled)
    area_unscaled = float(props_unscaled.Mass())
    area_metric_m2 = float(area_unscaled * (scale_mm_per_unit * 1e-3) ** 2)

    # 8. Tessellate Sewn Shell and Verify with OpenFOAM surfaceCheck
    linear_defl = 0.0005  # ~0.13 mm
    angular_defl = 0.03
    BRepMesh_IncrementalMesh(sewn_shell, linear_defl, False, angular_defl, True)

    all_verts = []
    all_tris = []
    v_offset = 0
    fexp = TopExp_Explorer(sewn_shell, TopAbs_FACE)
    while fexp.More():
        f = TopoDS.Face(fexp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        trsf = loc.Transformation()
        verts = np.array(
            [
                [
                    tri.Node(i).Transformed(trsf).X(),
                    tri.Node(i).Transformed(trsf).Y(),
                    tri.Node(i).Transformed(trsf).Z(),
                ]
                for i in range(1, tri.NbNodes() + 1)
            ],
            dtype=float,
        )
        tris = (
            np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=int)
            - 1
        )
        if f.Orientation() == TopAbs_REVERSED:
            tris = tris[:, [0, 2, 1]]
        all_verts.append(verts)
        all_tris.append(tris + v_offset)
        v_offset += len(verts)
        fexp.Next()

    V = np.vstack(all_verts)
    F = np.vstack(all_tris)
    mesh_unscaled = trimesh.Trimesh(vertices=V, faces=F, process=True)
    non_degen = mesh_unscaled.area_faces > 1e-14
    if not np.all(non_degen):
        mesh_unscaled.update_faces(non_degen)
        mesh_unscaled.remove_unreferenced_vertices()

    stl_unscaled_path = R / "vehicle_sidepod_shell_UNSCALED.stl"
    obj_unscaled_path = R / "vehicle_sidepod_shell_UNSCALED.obj"
    ply_unscaled_path = R / "vehicle_sidepod_shell_UNSCALED.ply"
    mesh_unscaled.export(stl_unscaled_path)
    mesh_unscaled.export(obj_unscaled_path)
    mesh_unscaled.export(ply_unscaled_path)

    mesh_metric = mesh_unscaled.copy()
    mesh_metric.vertices = mesh_metric.vertices * scale_mm_per_unit
    stl_metric_path = R / "vehicle_sidepod_shell_METRIC_mm.stl"
    obj_metric_path = R / "vehicle_sidepod_shell_METRIC_mm.obj"
    mesh_metric.export(stl_metric_path)
    mesh_metric.export(obj_metric_path)

    # Trimesh topological audit
    edge_counts = np.bincount(mesh_unscaled.edges_unique_inverse)
    n_boundary_edges = int((edge_counts == 1).sum())
    n_manifold_internal_edges = int((edge_counts == 2).sum())
    n_nonmanifold_edges = int((edge_counts > 2).sum())
    assert n_nonmanifold_edges == 0, f"Mesh has {n_nonmanifold_edges} non-manifold edges!"

    # Run OpenFOAM surfaceCheck -checkSelfIntersection on the metric STL
    of_cmd = (
        f"source {os.environ['SCAN2FLOW_OF_BASHRC']} && "
        f"surfaceCheck -checkSelfIntersection {stl_metric_path}"
    )
    proc = subprocess.run(["bash", "-c", of_cmd], capture_output=True, text=True, check=True)
    of_log = proc.stdout + "\n" + proc.stderr
    (R / "openfoam_surfaceCheck.log").write_text(of_log)

    assert "Surface has no illegal triangles." in of_log
    assert "connected to >2 faces : 0" in of_log
    assert "Number of unconnected parts : 1" in of_log
    assert "Number of zones (connected area with consistent normal) : 1" in of_log
    assert "Surface is not self-intersecting" in of_log

    # Write JSON summaries
    sewing_audit = {
        "source_upper_dir": str(upper_dir),
        "source_seam_extension_dir": str(seam_ext_dir),
        "source_seam_corner_dir": str(seam_corner_dir),
        "sewing_tolerance_unscaled": sewing_tol,
        "sewing_tolerance_metric_mm": float(sewing_tol * scale_mm_per_unit),
        "scale_mm_per_unit": scale_mm_per_unit,
        "scale_show_670_mm_per_unit": scale_show_670_mm_per_unit,
        "scale_midpoint_mm_per_unit": scale_midpoint_mm_per_unit,
        "scale_statistical_uncertainty_pct": scale_statistical_uncertainty_pct,
        "scale_uncertainty_pct": scale_uncertainty_pct,
        "scale_source": scale_source,
        "brep_topology": {
            "shells": 1,
            "faces": 11,
            "free_edges": len(free_edges),
            "contiguous_internal_edges": len(contiguous_edges),
            "multiple_nonmanifold_edges": len(multiple_edges),
            "degenerated_edges": len(degenerated_edges),
            "brepcheck_valid": True,
            "step_roundtrip_valid": True,
            "surface_area_unscaled_sq_units": area_unscaled,
            "surface_area_metric_m2": area_metric_m2,
        },
        "tessellation": {
            "linear_deflection_unscaled": linear_defl,
            "linear_deflection_metric_mm": float(linear_defl * scale_mm_per_unit),
            "angular_deflection_rad": angular_defl,
            "vertices": int(len(mesh_unscaled.vertices)),
            "triangles": int(len(mesh_unscaled.faces)),
            "boundary_edges": n_boundary_edges,
            "manifold_internal_edges": n_manifold_internal_edges,
            "nonmanifold_edges": n_nonmanifold_edges,
            "openfoam_surfaceCheck": {
                "illegal_triangles": 0,
                "orientable": True,
                "connected_zones": 1,
                "conflicting_orientations": 0,
                "self_intersections": 0,
                "open_boundary_edges": n_boundary_edges,
            },
        },
        "free_boundaries": free_edge_records,
    }
    (R / "sewing_audit.json").write_text(json.dumps(sewing_audit, indent=2))
    (R / "free_boundaries.json").write_text(json.dumps(free_edge_records, indent=2))

    # Write SHA-256 manifest
    deliverables = [
        "sidepod_flank_reconciled_UNSCALED.step",
        "vehicle_sidepod_shell_UNSCALED.step",
        "vehicle_sidepod_shell_METRIC_mm.step",
        "free_boundaries_UNSCALED.step",
        "free_boundaries_METRIC_mm.step",
        "vehicle_sidepod_shell_UNSCALED.stl",
        "vehicle_sidepod_shell_UNSCALED.obj",
        "vehicle_sidepod_shell_UNSCALED.ply",
        "vehicle_sidepod_shell_METRIC_mm.stl",
        "vehicle_sidepod_shell_METRIC_mm.obj",
        "sewing_audit.json",
        "free_boundaries.json",
        "openfoam_surfaceCheck.log",
    ]
    hashes = {
        fn: hashlib.sha256((R / fn).read_bytes()).hexdigest()
        for fn in deliverables
        if (R / fn).exists()
    }
    (R / "deliverables_sha256.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps(sewing_audit["brep_topology"], indent=2))
    print(json.dumps(sewing_audit["tessellation"], indent=2))


if __name__ == "__main__":
    main()
