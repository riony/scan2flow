"""Amendment 1 corrected implementation; tested recipes still failed feature gates.

Caller-provided mesh and feature segments; no automatic retries or acceptance.
"""

import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

from .control_common import distance_segments, save, write_off
from .mesh_extract import extract


def run(mesh, segments, candidate, output_prefix):
    """Write voxel, closed and final OFF checkpoints plus recipe metrics.

    Candidate keys: pitch, erode_voxels, protected_filter, smooth_iterations, id.
    Units must match caller geometry; historical fixed strips are .013/.026.
    """
    out = Path(output_prefix)
    pitch = candidate["pitch"]
    t0 = time.monotonic()
    m = mesh
    dims = np.ceil(m.extents / pitch).astype(np.int64) + 13
    if np.prod(dims) > 750000000:
        raise RuntimeError("dense allocation cap")
    vox = m.voxelized(pitch).fill()
    shape = np.array(vox.shape)
    if np.prod(shape) > 750000000:
        raise RuntimeError("dense grid cap")
    raw = extract(vox, out.with_name(out.name + "_voxel.off"))
    if candidate["erode_voxels"]:
        mat = ndimage.binary_erosion(vox.matrix, iterations=candidate["erode_voxels"])
        vox = trimesh.voxel.VoxelGrid(mat, transform=vox.transform)
    w = extract(vox, out.with_name(out.name + "_closed.off"))
    # Keep every component. Component rejection belongs to the gate.
    L = trimesh.smoothing.laplacian_calculation(w)
    v = w.vertices.copy()
    weight = (
        np.clip((distance_segments(v, segments) - 0.013) / 0.013, 0, 1)
        if candidate["protected_filter"]
        else np.ones(len(v))
    )
    initial = v.copy()
    for i in range(candidate["smooth_iterations"]):
        delta = L.dot(v) - v
        v += (0.5 if i % 2 == 0 else -0.5) * weight[:, None] * delta
    w.vertices = v
    write_off(w, out.with_suffix(".off"))
    save(
        out.with_name(out.name + "_dense.json"),
        {
            "candidate": candidate,
            "pitch": pitch,
            "dense_shape": shape.tolist(),
            "cells": int(np.prod(shape)),
            "max_displacement_in_zero_mask": float(
                np.linalg.norm(v[weight == 0] - initial[weight == 0], axis=1).max()
            )
            if np.any(weight == 0)
            else 0,
            "components_kept": len(w.split(only_watertight=False)),
            "seconds": time.monotonic() - t0,
            "source_projection": False,
        },
    )
    print("complete", candidate["id"], len(w.faces), flush=True)
