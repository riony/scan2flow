"""Read-only mesh inspection. No implicit repair, unit conversion or solver run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


def boundary_counts(edges: np.ndarray) -> dict[str, int]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    unseen = set(adjacency)
    loops = chains = branched = 0
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbour in adjacency[vertex]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        degrees = [len(adjacency[v]) for v in component]
        if all(d == 2 for d in degrees):
            loops += 1
        elif degrees.count(1) == 2 and all(d in (1, 2) for d in degrees):
            chains += 1
        else:
            branched += 1
    return {
        "boundary_loops": loops,
        "boundary_open_chains": chains,
        "boundary_branched_components": branched,
    }


def inspect_mesh(path: Path) -> dict:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("input must contain a nonempty triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh contains non-finite coordinates")
    original_vertices = len(mesh.vertices)
    # STL stores triangle-local vertices. Merge coordinates for incidence analysis
    # only; the source file is never changed. Precision is explicit in the output.
    mesh.merge_vertices(digits_vertex=12)
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    area_limit = np.finfo(float).eps * max(float(np.dot(mesh.extents, mesh.extents)), 1e-30)
    return {
        "file": path.name,
        "units": "input units; no scale inferred",
        "input_vertices": original_vertices,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "coordinate_merge_decimal_places": 12,
        "bounds": mesh.bounds.tolist(),
        "extents": mesh.extents.tolist(),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "degenerate_faces": int(np.sum(mesh.area_faces <= area_limit)),
        **boundary_counts(boundary),
        "limitations": "No self-intersection, vertex-manifold, physical-shape or CFD-readiness certification.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Print mesh topology and bounds as JSON")
    inspect.add_argument("mesh", type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_mesh(args.mesh)
    except (OSError, ValueError, TypeError, IndexError) as exc:
        print(f"scan2flow: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
