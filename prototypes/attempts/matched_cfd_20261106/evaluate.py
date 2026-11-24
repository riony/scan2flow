"""Historical v2412 matched-control evidence parser. Never invokes OpenFOAM.

The 24-rank layout, bounds, reference values and thresholds belong to the saved
control protocol; this is not a general-purpose or production acceptance gate.
"""

import math
import re

import numpy as np

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PATCHES = {"inlet", "outlet", "ground", "top", "sides", "car"}
FATAL = r"FOAM FATAL|Segmentation fault|MPI_ABORT|Floating point exception(?! trapping)|\bDIVERGED\b|(?<![\w])[-+]?(?:nan|inf(?:inity)?)(?![\w])"


def one(text, pattern, cast=float):
    values = re.findall(pattern, text, re.MULTILINE)
    if len(values) != 1:
        return None
    try:
        value = cast(values[0])
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def complete_log(text, returncode, ranks):
    versions = set(re.findall(r"(?:Version:\s*|OPENFOAM=)v?(\d+)\b", text))
    processes = one(text, r"^nProcs\s*:\s*(\d+)\s*$", int)
    return {
        "returncode_zero": type(returncode) is int and returncode == 0,
        "complete_end": bool(re.search(r"^End\s*$", text, re.MULTILINE)),
        "version_2412": versions == {"2412"},
        "ranks_expected": processes == ranks,
        "no_fatal_or_nonfinite": re.search(FATAL, text, re.IGNORECASE) is None,
    }


def parse_mesh(text, returncode=0):
    gates = complete_log(text, returncode, 24)
    metrics = {
        "cells": one(text, r"^\s*cells:\s*(\d+)\s*$", int),
        "faces": one(text, r"^\s*faces:\s*(\d+)\s*$", int),
        "points": one(text, r"^\s*points:\s*(\d+)\s*$", int),
        "regions": one(text, r"Number of regions:\s*(\d+)\b", int),
        "min_volume_m3": one(text, r"Min volume\s*=\s*(" + FLOAT + ")"),
        "total_volume_m3": one(text, r"Total volume\s*=\s*(" + FLOAT + ")"),
        "max_nonorthogonality": one(text, r"Mesh non-orthogonality Max:\s*(" + FLOAT + ")"),
        "max_skewness": one(text, r"Max skewness\s*=\s*(" + FLOAT + ")"),
    }
    rows = re.findall(
        r"^\s*(inlet|outlet|ground|top|sides|car|symmetry)\s+(\d+)\s+(\d+)\s+([^\n]+)$",
        text,
        re.MULTILINE,
    )
    patches = {
        name: {"faces": int(faces), "points": int(points), "topology": topology.strip()}
        for name, faces, points, topology in rows
    }
    bounds = re.findall(r"Overall domain bounding box\s*\(([^)]+)\)\s*\(([^)]+)\)", text)
    try:
        bbox = (
            np.array([[float(v) for v in group.split()] for group in bounds[0]])
            if len(bounds) == 1
            else None
        )
        bbox_ok = (
            bbox is not None
            and bbox.shape == (2, 3)
            and np.allclose(bbox, [[-3.1, -3.5, 0], [9.4, 3.5, 3.5]], rtol=0, atol=1e-7)
        )
    except ValueError:
        bbox, bbox_ok = None, False
    gates.update(
        {
            "explicit_mesh_ok": bool(re.search(r"^Mesh OK\.\s*$", text, re.MULTILINE)),
            "zero_failed_checks": re.search(r"Failed\s+\d+\s+mesh checks|\*\*\*", text) is None,
            "all_metrics_present": all(v is not None for v in metrics.values()),
            "positive_cells_under_cap": metrics["cells"] is not None
            and 0 < metrics["cells"] <= 8000000,
            "one_fluid_region": metrics["regions"] == 1,
            "positive_volume": metrics["min_volume_m3"] is not None
            and metrics["min_volume_m3"] > 0,
            "volume_check_explicit": "Cell volumes OK." in text,
            "nonorth_gate": metrics["max_nonorthogonality"] is not None
            and metrics["max_nonorthogonality"] <= 65,
            "skew_gate": metrics["max_skewness"] is not None and metrics["max_skewness"] <= 4,
            "six_physical_patches": set(patches) == PATCHES and len(rows) == 6,
            "all_patches_have_faces": bool(patches)
            and all(v["faces"] > 0 for v in patches.values()),
            "patch_topology_ok": bool(patches)
            and all(v["topology"].startswith("ok ") for v in patches.values()),
            "full_domain_bounds": bool(bbox_ok),
        }
    )
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "patches": patches,
        "domain_bounds_m": bbox.tolist() if bbox is not None else None,
    }


def table(text):
    headers = [line[1:].split() for line in text.splitlines() if re.match(r"^#\s*Time\b", line)]
    assert len(headers) == 1 and len(set(headers[0])) == len(headers[0]), (
        "Missing/duplicate table header"
    )
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    values = np.asarray(rows, dtype=float)
    assert values.ndim == 2 and values.shape[1] == len(headers[0]) and np.isfinite(values).all()
    assert np.all(np.diff(values[:, 0]) > 0), "Duplicate or nonmonotonic force samples"
    return {name: values[:, i] for i, name in enumerate(headers[0])}


def force_metrics(text):
    data = table(text)
    assert {"Time", "Cd", "Cl"} <= data.keys()
    t = data["Time"]
    gates = {
        "full_3000_force_schedule": np.array_equal(t, np.arange(1, 3001)),
        "late_window_complete": np.array_equal(t[(t >= 2500) & (t <= 3000)], np.arange(2500, 3001)),
    }
    headers = {}
    for name, expected in [("magUInf", 40), ("lRef", 0.288), ("Aref", 0.11510)]:
        value = one(text, r"^#\s*" + name + r"\s*:\s*(" + FLOAT + r")\s*$")
        headers[name] = value
        gates[name + "_matches"] = value is not None and math.isclose(
            value, expected, rel_tol=1e-10, abs_tol=1e-12
        )
    for name, expected in [
        ("dragDir", [1, 0, 0]),
        ("liftDir", [0, 0, 1]),
        ("CofR", [0.5, 0, 0.15]),
    ]:
        found = re.findall(r"^#\s*" + name + r"\s*:\s*\(([^)]+)\)\s*$", text, re.MULTILINE)
        value = [float(x) for x in found[0].split()] if len(found) == 1 else []
        headers[name] = value
        gates[name + "_matches"] = len(value) == 3 and np.allclose(
            value, expected, rtol=0, atol=1e-10
        )
    means, late, changes = {}, {}, {}
    for name, limit in [("Cd", 0.002), ("Cl", 0.005)]:
        v = data[name]
        selections = [
            (t >= 2500) & (t <= 3000),
            (t >= 2501) & (t <= 2750),
            (t >= 2751) & (t <= 3000),
        ]
        values = [float(np.mean(v[m])) if m.any() else None for m in selections]
        means[name], late[name] = values[0], values[1:]
        changes[name] = (
            abs(values[2] - values[1]) if all(x is not None for x in values[1:]) else None
        )
        gates[name + "_steady"] = changes[name] is not None and changes[name] <= limit
    return {
        "passed": all(bool(v) for v in gates.values()),
        "gates": {k: bool(v) for k, v in gates.items()},
        "headers": headers,
        "samples": len(t),
        "means_2500_3000": means,
        "late_250_sample_means": late,
        "late_mean_abs_changes": changes,
        "range": {
            name: [float(data[name].min()), float(data[name].max())] for name in ("Cd", "Cl")
        },
        "history": {name: data[name].tolist() for name in ("Time", "Cd", "Cl")},
    }
