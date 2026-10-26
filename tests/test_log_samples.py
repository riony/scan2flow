"""Characterize legacy parser behavior without requiring an OpenFOAM install."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("legacy_foam", ROOT / "prototypes/pipeline/foam.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def test_failed_checkmesh_fixture():
    result = legacy.parse_checkmesh(ROOT / "fixtures", "checkMesh_sample.log")
    assert result == {
        "n_cells": 1281913.0,
        "max_nonortho": 64.83682758,
        "max_skewness": 5.750648743,
        "ok": 0.0,
    }


def test_legacy_truncated_log_is_not_a_safe_gate(tmp_path):
    # Deliberately record the migration defect, rather than treating it as success.
    assert legacy.parse_checkmesh(tmp_path) == {}
    (tmp_path / "log.checkMesh").write_text("Create mesh for time = 0\n")
    assert legacy.parse_checkmesh(tmp_path) == {"ok": 1.0}


def test_all_python_sources_compile_without_optional_runtimes():
    for directory in ("src", "prototypes"):
        for path in (ROOT / directory).rglob("*.py"):
            compile(path.read_text(), str(path), "exec")
