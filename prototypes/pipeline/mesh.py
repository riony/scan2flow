"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import foam
from .config import CaseConfig

# Project quality gates (plan.md §Specifications)
GATE_NONORTHO = 65.0
GATE_SKEWNESS = 4.0


class OrientationNotVerified(RuntimeError):
    pass


def _assert_orientation_verified(case: Path):
    """HARD gate: refuse to mesh until a human has confirmed the ingest orientation
    check (nose at -x, flow +x). The F1 and GT-R both silently ran REVERSED in June
    2026 — invalid Cd/Cl, full re-runs. A warning was not enough; this is structural.
    Verify with:  python -m pipeline.ingest verify <geometry_dir>"""
    for md in [case / "geometry" / "metadata.json", case.parent / "geometry" / "metadata.json"]:
        if md.exists():
            meta = json.loads(md.read_text())
            o = meta.get("orientation", {})
            if o and not o.get("verified", False):
                png = o.get("check_png", "geometry/_orientation_check.png")
                raise OrientationNotVerified(
                    f"{md}: orientation.verified is false.\n"
                    f"  1) view {png} — the nose MUST point to -x (left), flow is +x\n"
                    f"  2) then run: python -m pipeline.ingest verify {md.parent}\n"
                    f"  (silent-reversal guard; see scan2flow-ingest-orientation-bug)"
                )
            return
    print(
        "    [gate] no geometry/metadata.json found — orientation gate skipped "
        "(validation-body case?)"
    )


def mesh_case(case: Path, cfg: CaseConfig) -> dict[str, float]:
    """Mesh a single case directory; return checkMesh metrics and warn on gate breaches."""
    _assert_orientation_verified(case)
    print(f"  meshing {case} ...")
    foam.run("./Allmesh", case, log="log.Allmesh")
    q = foam.parse_checkmesh(case)

    no = q.get("max_nonortho")
    sk = q.get("max_skewness")
    cells = q.get("n_cells")
    print(
        f"    cells={cells:,.0f}" if cells else "    cells=?",
        f"maxNonOrtho={no}",
        f"maxSkewness={sk}",
    )
    if no is not None and no > GATE_NONORTHO:
        print(f"    [GATE] non-orthogonality {no} > {GATE_NONORTHO}")
    if sk is not None and sk > GATE_SKEWNESS:
        print(f"    [GATE] skewness {sk} > {GATE_SKEWNESS}")
    return q
