"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from .foam import of_bashrc


def _sh(cmd: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", f"source {of_bashrc()}; cd '{cwd}'; {cmd}"], capture_output=True, text=True
    )


def surface_metrics(stl: str | Path) -> dict:
    """OF surfaceCheck summary: closed?, #regions, illegal tris, min edge."""
    r = _sh(f"surfaceCheck '{Path(stl).resolve()}'", Path(stl).resolve().parent)
    out = r.stdout

    def grab(key, default=None):
        for ln in out.splitlines():
            if key in ln:
                return ln.strip()
        return default

    closed = "Surface is closed" in out and "not closed" not in out
    import re

    tris = re.search(r"Triangles\s*:\s*(\d+)\s*in\s*(\d+)", out)
    illegal = re.search(r"has (\d+) illegal", out)
    minedge = re.search(r"min ([\d.eE+-]+) for edge", out)
    return {
        "closed": closed,
        "triangles": int(tris.group(1)) if tris else None,
        "regions": int(tris.group(2)) if tris else None,
        "illegal": int(illegal.group(1)) if illegal else 0,
        "min_edge_m": float(minedge.group(1)) if minedge else None,
    }


def snappy_wrap(
    stl_in: str | Path,
    stl_out: str | Path,
    wrap_level: int = 3,
    nprocs: int = 24,
    base_cell: float | None = None,
    feature_snap: bool = True,
    work: str | Path | None = None,
    template: str | Path | None = None,
) -> dict:
    """Wrap a messy STL with snappy and extract the closed snapped shell -> stl_out.

    wrap_level : surface refinement level (cell at surface = base_cell / 2**level).
                 Lower => more gap-bridging / smoother; higher => more feature-faithful.
    base_cell  : background cell size; default = bbox diagonal / 60.
    Returns surface_metrics of the result plus timing.
    """
    import time

    t0 = time.time()
    stl_in = Path(stl_in).resolve()
    stl_out = Path(stl_out).resolve()
    if work is None or template is None:
        raise ValueError("Explicit work and template directories are required")
    work = Path(work)
    if work.exists():
        raise FileExistsError(f"Refusing to replace existing workspace: {work}")
    shutil.copytree(template, work)

    m = trimesh.load(stl_in, force="mesh")
    lo, hi = m.bounds
    L = float(np.linalg.norm(hi - lo))
    if base_cell is None:
        base_cell = L / 60.0
    # background block: pad the bbox so the body sits in free space
    pad = 0.5 * (hi - lo) + 0.1 * L
    bmin, bmax = lo - pad, hi + pad
    n = np.maximum(((bmax - bmin) / base_cell).astype(int), 4)
    # a seed point guaranteed OUTSIDE the body (a far corner of the padded block)
    seed = bmin + 0.05 * (bmax - bmin)

    tri = work / "constant" / "triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    for f in tri.glob("*"):
        f.unlink()
    shutil.copy(stl_in, tri / "car.stl")

    # blockMeshDict
    (work / "system" / "blockMeshDict").write_text(_block_dict(bmin, bmax, n))
    # snappyHexMeshDict (single region wrap, NO layers)
    (work / "system" / "snappyHexMeshDict").write_text(
        _snappy_wrap_dict(wrap_level, seed, feature_snap)
    )
    # surfaceFeatureExtractDict
    (work / "system" / "surfaceFeatureExtractDict").write_text(_sfe_dict())

    log = work / "log.wrap"
    steps = (
        "blockMesh",
        "surfaceFeatureExtract" if feature_snap else "true",
        "decomposePar -force",
        f"mpirun -np {nprocs} --oversubscribe snappyHexMesh -overwrite -parallel",
        "reconstructParMesh -constant -mergeTol 1e-6",
        # extract the wrapped car patch to a surface
        f"surfaceMeshExtract -patches '(car)' '{stl_out}'",
    )
    # need a numberOfSubdomains matching nprocs
    (work / "system" / "decomposeParDict").write_text(_decompose_dict(nprocs))
    full = " && ".join(s for s in steps if s != "true")
    r = _sh(
        f"{{ {full} ; }} > '{log}' 2>&1; echo RC=$?", work
    )  # group so redirect covers all steps
    rc = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "RC=?"

    res = {
        "rc": rc,
        "wrap_level": wrap_level,
        "base_cell_m": round(base_cell, 4),
        "surf_cell_mm": round(1000 * base_cell / 2**wrap_level, 2),
        "secs": round(time.time() - t0, 1),
        "out": str(stl_out),
        "work": str(work),
    }
    if stl_out.exists():
        res["result"] = surface_metrics(stl_out)
    else:
        res["error"] = "no output surface; see " + str(log)
    return res


def sdf_wrap(
    stl_in: str | Path,
    stl_out: str | Path,
    pitch: float | None = None,
    smooth_iter: int = 30,
    erode: int = 0,
) -> dict:
    """FAST closed-envelope wrap: voxel-fill -> marching cubes -> Taubin smooth.
    Seconds vs minutes for snappy_wrap, and closes EVERYTHING (gaps + interpenetration).
    The earlier 'voxel wrap is dead' verdict was for COARSE pitch (terracing wrecks
    layers); here a fine pitch + Taubin smoothing removes the stair-stepping. Still
    resolution-limited on thin features (a wing may merge into the body) -> always run
    the validation harness before trusting it for a validated Cd. Good for salvaging
    otherwise-unusable viz CAD into a smooth solid that at least meshes & is DDES-capable."""
    import time

    t0 = time.time()
    m = trimesh.load(stl_in, force="mesh")
    ext = m.bounds[1] - m.bounds[0]
    diag = float(np.linalg.norm(ext))
    pitch = pitch or diag / 450.0  # ~fine; keeps voxel count sane
    vox = m.voxelized(pitch).fill()
    if erode:  # cancel the ~half-voxel outward offset
        from scipy import ndimage

        mat = ndimage.binary_erosion(vox.matrix, iterations=erode)
        vox = trimesh.voxel.VoxelGrid(mat, transform=vox.transform)
    w = vox.marching_cubes
    w.apply_transform(vox.transform)  # voxel-index -> world coords (critical)
    comps = w.split(only_watertight=False)
    if len(comps):
        w = max(comps, key=lambda c: len(c.faces))  # largest shell, drop specks
    w.merge_vertices()
    w.update_faces(w.nondegenerate_faces())
    w.fix_normals()
    trimesh.smoothing.filter_taubin(w, iterations=smooth_iter)  # de-terrace, volume-preserving
    Path(stl_out).parent.mkdir(parents=True, exist_ok=True)
    w.export(stl_out)
    res = {
        "pitch_mm": round(1000 * pitch, 1),
        "smooth_iter": smooth_iter,
        "secs": round(time.time() - t0, 1),
        "out": str(stl_out),
        "result": surface_metrics(stl_out),
    }
    return res


def vtk_repair(
    stl_in: str | Path, stl_out: str | Path, hole_frac: float = 0.05, taubin_iter: int = 20
) -> dict:
    """Lighter surface repair (no snappy): VTK fill-holes + clean + Taubin smooth.
    Good for mostly-clean meshes with a few tears; does NOT resolve interpenetrating
    parts (use snappy_wrap for those). Taubin smoothing is ~volume-preserving so it
    won't shrink the body like Laplacian smoothing would."""
    import pyvista as pv

    m = pv.read(str(stl_in)).clean()
    b = np.array(m.bounds)
    diag = float(np.linalg.norm(b[1::2] - b[0::2]))
    m = m.fill_holes(hole_frac * diag)  # close tears up to frac*diag
    m = m.smooth_taubin(n_iter=taubin_iter, pass_band=0.05)  # de-noise, volume-preserving
    m = m.clean()
    m.extract_surface().triangulate().save(str(stl_out))
    return surface_metrics(stl_out)


def damage_surface(stl_in: str | Path, stl_out: str | Path, mode: str = "underbody") -> dict:
    """Make a clean surface 'dirty' to test that cleanup RECOVERS the true aero.
    mode='underbody': delete the bottom faces (mimics an open-bottom viz model);
    mode='holes': punch random holes (mimics a torn mesh)."""
    m = trimesh.load(stl_in, force="mesh")
    lo, hi = m.bounds
    if mode == "underbody":
        zc = lo[2] + 0.18 * (hi[2] - lo[2])
        keep = m.triangles_center[:, 2] > zc  # drop low faces -> open bottom
    else:
        rng = np.random.default_rng(0)
        keep = rng.random(len(m.faces)) > 0.12  # punch ~12% holes
    m.update_faces(keep)
    m.export(stl_out)
    return surface_metrics(stl_out)


# ----------------------------------------------------------------- dict writers
def _block_dict(bmin, bmax, n) -> str:
    v = lambda a: f"({a[0]:.4f} {a[1]:.4f} {a[2]:.4f})"
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
 ({bmin[0]:.4f} {bmin[1]:.4f} {bmin[2]:.4f}) ({bmax[0]:.4f} {bmin[1]:.4f} {bmin[2]:.4f})
 ({bmax[0]:.4f} {bmax[1]:.4f} {bmin[2]:.4f}) ({bmin[0]:.4f} {bmax[1]:.4f} {bmin[2]:.4f})
 ({bmin[0]:.4f} {bmin[1]:.4f} {bmax[2]:.4f}) ({bmax[0]:.4f} {bmin[1]:.4f} {bmax[2]:.4f})
 ({bmax[0]:.4f} {bmax[1]:.4f} {bmax[2]:.4f}) ({bmin[0]:.4f} {bmax[1]:.4f} {bmax[2]:.4f})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
edges ();
boundary ( domain {{ type patch; faces ((0 3 2 1)(4 5 6 7)(0 1 5 4)(2 3 7 6)(0 4 7 3)(1 2 6 5)); }} );
mergePatchPairs ();
"""


def _snappy_wrap_dict(level, seed, feature_snap) -> str:
    fs = "true" if feature_snap else "false"
    feat = '{ file "car.eMesh"; level %d; }' % level if feature_snap else ""
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true; snap true; addLayers false;
geometry {{ car {{ type triSurfaceMesh; file "car.stl"; }} }}
castellatedMeshControls
{{
    maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 0;
    nCellsBetweenLevels 3; resolveFeatureAngle 30;
    features ( {feat} );
    refinementSurfaces {{ car {{ level ({level} {level}); patchInfo {{ type wall; }} }} }}
    refinementRegions {{ }}
    locationInMesh ({seed[0]:.4f} {seed[1]:.4f} {seed[2]:.4f});
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8;
    nFeatureSnapIter 12; implicitFeatureSnap false; explicitFeatureSnap {fs};
    multiRegionFeatureSnap false;
}}
addLayersControls {{ relativeSizes true; layers {{}}; expansionRatio 1.2;
    finalLayerThickness 0.3; minThickness 0.1; nGrow 0; }}
meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1;
    minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01;
    minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
}}
mergeTolerance 1e-6;
"""


def _sfe_dict() -> str:
    return """FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
car.stl { extractionMethod extractFromSurface;
  extractFromSurfaceCoeffs { includedAngle 150; } writeObj no; }
"""


def _decompose_dict(n) -> str:
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object decomposeParDict; }}
numberOfSubdomains {n}; method scotch;
"""
