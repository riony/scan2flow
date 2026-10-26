"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh


def load_full(source):
    """Concatenate the 3 half parts and mirror across y=0 to a full-car point soup."""
    parts = [trimesh.load(Path(source) / f"{f}.stl") for f in ("body", "front_wheel", "rear_wheel")]
    half = trimesh.util.concatenate(parts)
    mir = half.copy()
    mir.apply_transform(np.diag([1, -1, 1, 1.0]))
    return trimesh.util.concatenate([half, mir]), half


def metrics(m, tag):
    print(
        f"  [{tag}] watertight={m.is_watertight} bodies={m.body_count} "
        f"euler={m.euler_number} tris={len(m.faces)} "
        f"bbox={np.round(m.bounds, 3).tolist()}"
    )


def method_boolean(full):
    print("\n== method 1: boolean union (manifold3d) ==")
    try:
        t = time.time()
        u = trimesh.boolean.union([full], engine="manifold")
        print(f"  union ok in {time.time() - t:.1f}s")
        metrics(u, "boolean")
        return u
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return None


def method_wrap(full, pitch):
    print(f"\n== method 2: voxel shrink-wrap (pitch={pitch * 1000:.0f} mm) ==")
    t = time.time()
    vox = full.voxelized(pitch).fill()
    wrapped = vox.marching_cubes
    wrapped.apply_transform(vox.transform)  # map voxel indices -> world coordinates
    # keep only substantial shells (drop tiny spurious voxel specks)
    comps = wrapped.split(only_watertight=False)
    if len(comps) > 1:
        comps = [c for c in comps if c.extents.max() > 0.2]
        wrapped = trimesh.util.concatenate(comps)
    wrapped.merge_vertices()
    wrapped.update_faces(wrapped.nondegenerate_faces())
    wrapped.fix_normals()
    trimesh.smoothing.filter_taubin(wrapped, iterations=12)
    print(f"  wrapped in {time.time() - t:.1f}s")
    metrics(wrapped, "wrap-full")
    return wrapped


def to_half(full_mesh):
    """Slice the repaired full car back to the y>=0 half (open at the symmetry plane)."""
    half = full_mesh.slice_plane([0, 0, 0], [0, 1, 0])
    return half


def frontal_area(mesh):
    """Raster silhouette (x-projection) of the half body -> full frontal area."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for v in mesh.vertices:
        pts.InsertNextPoint(*v)
    cells = vtk.vtkCellArray()
    for f in mesh.faces:
        cells.InsertNextCell(3)
        [cells.InsertCellPoint(int(i)) for i in f]
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    b = pd.GetBounds()
    N, PS = 2000, 0.8
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
    rw.SetSize(N, N)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetParallelScale(PS)
    cam.SetFocalPoint(b[1] + 5, 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))
    cam.SetPosition(b[1] + 15, 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))
    cam.SetViewUp(0, 0, 1)
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.Update()
    arr = vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(N, N, -1)
    half = int((arr[:, :, 0] > 40).sum()) * (2 * PS / N) ** 2
    return 2 * half


def main(source, output, pitch):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    full, half_raw = load_full(source)
    metrics(full, "raw full (mirrored)")
    method_boolean(full)  # expected to fail on this input
    wrapped_full = method_wrap(full, pitch)
    half = to_half(wrapped_full)
    half.merge_vertices()
    out = output / "wrapped.stl"
    # name the solid 'car' for snappy/cfMesh
    half.export(out, file_type="stl_ascii")
    txt = out.read_text().split("\n", 1)
    out.write_text("solid car\n" + txt[1].replace("endsolid", "endsolid car"))
    print(f"\nwrapped half-body -> {out}")
    print(f"  watertight(full)={wrapped_full.is_watertight}  half tris={len(half.faces)}")
    print(f"  frontal area = {frontal_area(half):.3f} m^2 (target ~2.16)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--pitch", type=float, required=True)
    args = ap.parse_args()
    main(args.source, args.output, args.pitch)
