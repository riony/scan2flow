"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------- helpers
def _raster_frontal_area(mesh, n: int = 2500, parallel_scale: float = 1.6) -> float:
    """Frontal area [m^2] of the FULL car by raster silhouette along +x (validated
    against DrivAer: 2.16 vs documented 2.17 m^2). Expects a full (not half) mesh."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for v in mesh.vertices:
        pts.InsertNextPoint(*v)
    cells = vtk.vtkCellArray()
    for f in mesh.faces:
        cells.InsertNextCell(3)
        for i in f:
            cells.InsertCellPoint(int(i))
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    b = pd.GetBounds()
    yc, zc = 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(pd)
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(1, 1, 1)
    a.GetProperty().SetLighting(False)
    ren = vtk.vtkRenderer()
    ren.AddActor(a)
    ren.SetBackground(0, 0, 0)
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(n, n)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetParallelScale(parallel_scale)
    cam.SetFocalPoint(b[1] + 5, yc, zc)
    cam.SetPosition(b[1] + 15, yc, zc)
    cam.SetViewUp(0, 0, 1)
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    arr = vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(n, n, -1)
    return float((arr[:, :, 0] > 40).sum()) * (2 * parallel_scale / n) ** 2


def _repair_lite(mesh, speck_frac: float = 0.04):
    """Cheap, scalable robustness pass: weld vertices, drop degenerate faces, and
    remove tiny disconnected specks. Component analysis is done with face-index
    labels (scipy sparse) — NOT mesh.split(), which builds one Trimesh per shell and
    is pathologically slow on models with thousands of shells (e.g. Prius: 6k shells)."""
    import trimesh

    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    adj = mesh.face_adjacency
    n = len(mesh.faces)
    if len(adj) == 0 or n == 0:
        return mesh
    # connected components over the face-adjacency graph -> label per face
    labels = trimesh.graph.connected_component_labels(adj, node_count=n)
    verts = mesh.vertices
    faces = mesh.faces
    # per-component bounding-box diagonal, vectorised
    keep_face = np.ones(n, dtype=bool)
    uniq, inv = np.unique(labels, return_inverse=True)
    if len(uniq) > 1:
        sizes = np.zeros(len(uniq))
        for li in range(len(uniq)):
            fmask = inv == li
            vid = faces[fmask].ravel()
            ext = verts[vid].max(axis=0) - verts[vid].min(axis=0)
            sizes[li] = float(np.linalg.norm(ext))
        thresh = speck_frac * sizes.max()
        small = set(np.where(sizes < thresh)[0])
        keep_face = np.array([inv[i] not in small for i in range(n)])
        mesh.update_faces(keep_face)
        mesh.remove_unreferenced_vertices()
    return mesh


def _orientation_check_render(mesh, png: Path) -> bool:
    """Silent-reversal guard. Renders the oriented car in side profile with the flow
    direction marked, so the nose can be confirmed at the -x (upstream) end BEFORE
    meshing. ingest only aligns the nose-tail AXIS, not its direction — a wrong source
    flow-axis SIGN silently meshes the car backwards (this is exactly how F1 & GT-R
    ran rear-first). Returns True if the render was written."""
    import os

    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(1500, 560))
    p.set_background("white")
    p.add_mesh(pv.wrap(mesh), color="#8a8a8a")
    p.view_xz()  # +x renders to the RIGHT
    p.camera.zoom(1.4)
    p.add_text(
        "FLOW +x  ===>      nose MUST be at the LEFT (-x, upstream)",
        position="upper_left",
        font_size=12,
        color="red",
    )
    p.screenshot(str(png))
    p.close()
    return True


# ----------------------------------------------------------------------- main
def ingest(
    cad: str | Path,
    slug: str,
    length_m: float,
    up: str = "z",
    flow: str = "x",
    out_root: Path | None = None,
) -> dict:
    """Ingest one car model. `length_m` = real vehicle length (sets the scale).
    `up`/`flow` describe the SOURCE model's axes ('x','y','z', optionally '-z' etc.)."""
    import trimesh

    cad = Path(cad)
    if out_root is None:
        raise ValueError("out_root must be an explicit disposable output directory")
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=False)

    import time

    t0 = time.time()

    def _log(s):
        print(f"[ingest +{time.time() - t0:5.1f}s] {s}", flush=True)

    mesh = trimesh.load(str(cad), force="mesh")
    n_raw = len(mesh.faces)
    _log(f"loaded {n_raw} faces")
    mesh = _repair_lite(mesh)
    _log(f"repair-lite -> {len(mesh.faces)} faces")

    # --- orient: map source (flow, up) -> (+x, +z) ------------------------------
    ax = {
        "x": [1, 0, 0],
        "y": [0, 1, 0],
        "z": [0, 0, 1],
        "-x": [-1, 0, 0],
        "-y": [0, -1, 0],
        "-z": [0, 0, -1],
    }
    f, u = np.array(ax[flow], float), np.array(ax[up], float)
    s = np.cross(u, f)  # side axis (right-handed)
    R = np.eye(4)
    R[:3, :3] = np.column_stack([f, s, u]).T  # rows: new x,y,z
    mesh.apply_transform(R)

    # --- scale to real length ----------------------------------------------------
    ext = mesh.bounds[1] - mesh.bounds[0]
    scale = length_m / ext[0]
    mesh.apply_scale(scale)

    # --- centre laterally, seat on ground, nose at x=0 ---------------------------
    b = mesh.bounds
    mesh.apply_translation([-b[0][0], -0.5 * (b[0][1] + b[1][1]), -b[0][2]])
    b = mesh.bounds

    _log(
        f"oriented+scaled: L={b[1][0] - b[0][0]:.2f} W={b[1][1] - b[0][1]:.2f} H={b[1][2] - b[0][2]:.2f}"
    )

    # --- frontal area (full car) --------------------------------------------------
    area_full = _raster_frontal_area(mesh)
    _log(f"frontal area {area_full:.3f} m^2")

    # --- orientation sanity gate (silent-reversal guard) --------------------------
    check_png = out / "_orientation_check.png"
    try:
        _orientation_check_render(mesh, check_png)
        _log(f"orientation check -> {check_png.name}  (VERIFY nose is at -x / left)")
    except Exception as e:  # never let the render block ingest
        _log(f"orientation render failed: {e}")
        check_png = None

    # --- half-cut at y=0 for the symmetric half-body pipeline ---------------------
    half = mesh.slice_plane([0, 0, 0], [0, 1, 0])  # keep y >= 0
    _log(f"half-cut -> {len(half.faces)} faces")

    car_stl = out / "car.stl"
    half.export(car_stl, file_type="stl_ascii")
    txt = car_stl.read_text().split("\n", 1)
    car_stl.write_text("solid car\n" + txt[1].replace("endsolid", "endsolid car"))
    full_stl = out / "car_full.stl"
    mesh.export(full_stl)

    meta = {
        "slug": slug,
        "source_file": str(cad),
        "license": "CHECK-SOURCE",
        "raw_faces": n_raw,
        "clean_faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "shells": int(mesh.body_count),
        "scale_applied": scale,
        "length_m": round(float(b[1][0] - b[0][0]), 4),
        "width_m": round(float(b[1][1] - b[0][1]), 4),
        "height_m": round(float(b[1][2] - b[0][2]), 4),
        "frontal_area_m2": round(area_full, 4),
        "aref_half_m2": round(area_full / 2, 4),
        "quality_flag": "ok" if len(mesh.faces) > 10000 else "needs_manual",
        "orientation": {
            "declared_flow_axis": flow,
            "declared_up_axis": up,
            "convention": "nose placed at x_min; car must FACE -x to oppose the +x solver inflow",
            "check_render": (check_png.name if check_png else None),
            "verified": False,
            "REVIEW": "open _orientation_check.png — nose must be at the LEFT (-x) end. "
            "If it faces +x, re-ingest with the opposite --flow sign (e.g. -x). "
            "This guard exists because F1 & GT-R silently ran reversed.",
        },
        "outputs": {"half": str(car_stl), "full": str(full_stl)},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(
        f"\n*** ORIENTATION CHECK: open geometry/clean/{slug}/_orientation_check.png and "
        f"confirm the nose is at the LEFT (-x / upstream) end BEFORE meshing. "
        f"If it faces +x the car will run REVERSED — re-ingest with --flow flipped. ***\n"
    )
    return meta


def verify_orientation(geom_dir):
    """Human sign-off on the orientation check: flips metadata orientation.verified to
    true — the HARD mesh gate in pipeline/mesh.py requires it before any meshing.
    Usage: python -m pipeline.ingest verify geometry/clean/<slug>"""
    md = Path(geom_dir) / "metadata.json"
    meta = json.loads(md.read_text())
    o = meta.setdefault("orientation", {})
    o["verified"] = True
    from datetime import date

    o["verified_on"] = date.today().isoformat()
    md.write_text(json.dumps(meta, indent=2))
    print(f"orientation VERIFIED -> {md}")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "verify":
        verify_orientation(_sys.argv[2])
        raise SystemExit(0)
    ap = argparse.ArgumentParser(description="ingest an internet car model")
    ap.add_argument("cad")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--length", type=float, required=True, help="real vehicle length [m]")
    ap.add_argument("--up", default="z", help="source model up axis (default z)")
    ap.add_argument("--flow", default="x", help="source model nose axis (default x)")
    ap.add_argument("--out-root", type=Path, required=True)
    a = ap.parse_args()
    ingest(a.cad, a.slug, a.length, up=a.up, flow=a.flow, out_root=a.out_root)
