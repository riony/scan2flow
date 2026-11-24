"""Evidence-only amended checkMesh gate. Never invokes OpenFOAM or changes a case."""

import hashlib
import json
import math
import re
import shlex
from pathlib import Path

import numpy as np

from . import evaluate as legacy

F = legacy.FLOAT
QUALITY = {
    "nonorthogonality": (rf"non-orthogonality >\s*({F})\s+degrees\s*:\s*(\d+)", [65.0]),
    "pyramid_volume": (rf"faces with face pyramid volume <\s*({F})\s*:\s*(\d+)", [1e-13]),
    "tet_quality": (rf"faces with face-decomposition tet quality <\s*({F})\s*:\s*(\d+)", [1e-15]),
    "face_concavity": (rf"faces with concavity >\s*({F})\s+degrees\s*:\s*(\d+)", [80.0]),
    "skewness": (
        rf"faces with skewness >\s*({F})\s*\(internal\) or\s*({F})\s*\(boundary\)\s*:\s*(\d+)",
        [4.0, 4.0],
    ),
    "face_weight": (
        rf"faces with interpolation weights \(0\.\.1\)\s*<\s*({F})\s*:\s*(\d+)",
        [0.02],
    ),
    "volume_ratio": (rf"faces with volume ratio of neighbour cells <\s*({F})\s*:\s*(\d+)", [0.01]),
    "face_twist": (rf"faces with face twist <\s*({F})\s*:\s*(\d+)", [0.02]),
    "cell_determinant": (rf"faces on cells with determinant <\s*({F})\s*:\s*(\d+)", [0.001]),
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inputs(log_path, command_record, calibration, diagnostic=False):
    path = Path(log_path).resolve()
    record = (
        json.loads(Path(command_record).read_text())
        if isinstance(command_record, (str, Path))
        else command_record
    )
    text = path.read_text(errors="replace")
    argv = record["argv"]
    assert isinstance(argv, list) and all(isinstance(v, str) for v in argv)
    positions = [i for i, token in enumerate(argv) if Path(token).name == "checkMesh"]
    assert len(positions) == 1 and record["name"] == "checkMesh"
    args = argv[positions[0] :]
    printed = re.findall(r"^Exec\s*:\s*(.+)$", text, re.MULTILINE)
    printed_args = shlex.split(printed[0]) if len(printed) == 1 else []
    required = (
        {"-constant", "-allGeometry"}
        if diagnostic
        else {"-constant", "-allTopology", "-meshQuality"}
    )
    forbidden = {"-meshQuality"} if diagnostic else {"-allGeometry", "-noTopology"}
    forbidden |= {"-allRegions", "-region", "-time", "-latestTime"}
    gates = legacy.complete_log(text, record["returncode"], 1 if calibration else 24)
    gates.update(
        {
            "record_log_matches": Path(record["log"]).resolve() == path,
            "required_invocation": required <= set(args) and not (forbidden & set(args)),
            "logged_invocation_matches": bool(printed_args)
            and Path(printed_args[0]).name == "checkMesh"
            and printed_args[1:] == args[1:],
            "parallel_matches_scope": ("-parallel" not in args)
            if calibration
            else ("-parallel" in args),
            "single_constant_mesh": re.findall(r"^Time = (\S+)\s*$", text, re.MULTILINE)
            == ["constant"],
        }
    )
    cases = re.findall(r"^Case\s*:\s*(.+)$", text, re.MULTILINE)
    case_arg = args[args.index("-case") + 1] if args.count("-case") == 1 else None
    gates["case_matches_command"] = (
        case_arg is not None
        and len(cases) == 1
        and Path(cases[0]).resolve() == Path(case_arg).resolve()
    )
    return path, record, text, gates


def _calibration_checks(text, parsed):
    """Retained serial control: numeric gates apply, matched-car layout does not."""
    checks = dict(parsed["gates"])
    for key in (
        "ranks_expected",
        "six_physical_patches",
        "all_patches_have_faces",
        "patch_topology_ok",
        "full_domain_bounds",
        "positive_cells_under_cap",
    ):
        checks.pop(key)
    section = re.search(
        r"Checking patch topology[^\n]*\n(.*?)(?:\n\s*Checking |\Z)", text, re.DOTALL
    )
    rows = re.findall(
        r"^\s*([A-Za-z_]\w*)\s+(\d+)\s+(\d+)\s+([^\n]+)$",
        section[1] if section else "",
        re.MULTILINE,
    )
    bbox = parsed["domain_bounds_m"]
    checks.update(
        {
            "calibration_patch_rows_present": bool(rows) and len({r[0] for r in rows}) == len(rows),
            "calibration_patches_positive": bool(rows)
            and all(int(r[1]) > 0 and int(r[2]) > 0 for r in rows),
            "calibration_patch_topology_ok": bool(rows)
            and all(r[3].startswith("ok ") for r in rows),
            "calibration_domain_finite_3d": bbox is not None
            and np.shape(bbox) == (2, 3)
            and bool(np.isfinite(bbox).all())
            and bool(np.all(np.array(bbox)[1] > np.array(bbox)[0])),
            "positive_cells": parsed["metrics"]["cells"] is not None
            and parsed["metrics"]["cells"] > 0,
        }
    )
    return checks, [
        {"name": n, "faces": int(f), "points": int(p), "topology": t} for n, f, p, t in rows
    ]


def evaluate_check(log_path, command_record, calibration=False):
    """Primary gate; command_record is a dict or JSON path. No meshQuality defaults."""
    try:
        path, record, text, invocation = _inputs(log_path, command_record, calibration)
        parsed = legacy.parse_mesh(text, record["returncode"])
        checks = dict(parsed["gates"])
        patches = parsed["patches"]
        if calibration:
            checks, patches = _calibration_checks(text, parsed)
            if "expected_patches" in record:
                checks["calibration_patch_identity"] = {p["name"] for p in patches} == set(
                    record["expected_patches"]
                )
        checks.update(invocation)
        quality = {}
        for name, (pattern, expected) in QUALITY.items():
            rows = re.findall(r"^\s*" + pattern + r"\s*$", text, re.MULTILINE)
            present = len(rows) == 1
            thresholds = [float(v) for v in rows[0][:-1]] if present else []
            count = int(rows[0][-1]) if present else None
            valid = (
                present
                and count == 0
                and len(thresholds) == len(expected)
                and all(
                    math.isclose(a, b, rel_tol=1e-10, abs_tol=0)
                    for a, b in zip(thresholds, expected)
                )
            )
            quality[name] = {
                "thresholds": thresholds,
                "expected": expected,
                "new_error_faces": count,
                "passed": valid,
            }
            checks["quality_" + name] = valid
        metrics = parsed["metrics"]
        checks.update(
            {
                "mesh_quality_enabled": text.count("Enabling user-defined geometry checks.") == 1,
                "all_topology_enabled": text.count(
                    "Enabling all (cell, face, edge, point) topology checks."
                )
                == 1,
                "one_quality_block": len(
                    re.findall(r"^Checking faces in error\s*:", text, re.MULTILINE)
                )
                == 1,
                "one_mesh_ok": len(re.findall(r"^Mesh OK\.\s*$", text, re.MULTILINE)) == 1,
                "positive_counts_and_total_volume": all(
                    metrics[k] is not None and metrics[k] > 0
                    for k in ("cells", "faces", "points", "total_volume_m3")
                ),
                "nonnegative_quality_metrics": all(
                    metrics[k] is not None and metrics[k] >= 0
                    for k in ("max_nonorthogonality", "max_skewness")
                ),
            }
        )
        return {
            "kind": "primary_mesh_gate",
            "calibration": calibration,
            "passed": all(bool(v) for v in checks.values()),
            "gates": checks,
            "metrics": metrics,
            "patches": patches,
            "domain_bounds_m": parsed["domain_bounds_m"],
            "mesh_quality_checks": quality,
            "command": record,
            "log": str(path),
            "log_sha256": _sha(path),
            "parser_sha256": _sha(__file__),
            "legacy_parser_sha256": _sha(legacy.__file__),
        }
    except (AssertionError, KeyError, IndexError, ValueError, TypeError, OSError) as exc:
        return {
            "kind": "primary_mesh_gate",
            "calibration": calibration,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def evaluate_diagnostic(log_path, command_record, calibration=False):
    """Diagnostic completion, not geometric acceptance. Expected failed checks are retained."""
    try:
        path, record, text, gates = _inputs(log_path, command_record, calibration, diagnostic=True)
        counts = re.findall(
            r"Concave cells \(using face planes\) found,\s*number of cells:\s*(\d+)", text
        )
        concave_ok = len(re.findall(r"^\s*Concave cell check OK\.\s*$", text, re.MULTILINE))
        count = (
            int(counts[0])
            if len(counts) == 1 and not concave_ok
            else (0 if not counts and concave_ok == 1 else None)
        )
        failed = re.findall(r"^Failed (\d+) mesh checks\.\s*$", text, re.MULTILINE)
        okay = len(re.findall(r"^Mesh OK\.\s*$", text, re.MULTILINE))
        gates.update(
            {
                "all_geometry_enabled": text.count("Enabling all geometry checks.") == 1,
                "concave_cell_result_present": count is not None,
                "one_final_disposition": (len(failed) == 1 and not okay)
                or (not failed and okay == 1),
            }
        )
        return {
            "kind": "all_geometry_diagnostic",
            "calibration": calibration,
            "completed": all(bool(v) for v in gates.values()),
            "gates": gates,
            "concave_cells": count,
            "reported_failed_checks": int(failed[0])
            if len(failed) == 1
            else 0
            if okay == 1
            else None,
            "acceptance_role": "diagnostic_only; never a primary pass",
            "command": record,
            "log": str(path),
            "log_sha256": _sha(path),
        }
    except (AssertionError, KeyError, IndexError, ValueError, TypeError, OSError) as exc:
        return {
            "kind": "all_geometry_diagnostic",
            "calibration": calibration,
            "completed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
