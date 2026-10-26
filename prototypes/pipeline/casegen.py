"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import CaseConfig, case_definition


def _set_ground_slip(case: Path) -> None:
    """Swap the moving-ground BC to slip in the 0.orig fields."""
    repl = {
        "U": (
            "    ground\n    {\n        type            fixedValue;     // moving road, matches freestream\n"
            "        value           uniform $UinfVec;\n    }",
            "    ground\n    {\n        type            slip;\n    }",
        ),
    }
    for fld, (old, new) in repl.items():
        f = case / "0.orig" / fld
        txt = f.read_text()
        if old in txt:
            f.write_text(txt.replace(old, new))
    # k/omega/nut wall-function ground -> slip
    for fld in ("k", "omega", "nut"):
        f = case / "0.orig" / fld
        lines, out, skip = f.read_text().splitlines(), [], 0
        for i, ln in enumerate(lines):
            if ln.strip() == "ground" and i + 1 < len(lines) and lines[i + 1].strip() == "{":
                out += ["    ground", "    {", "        type            slip;", "    }"]
                skip = 1
                continue
            if skip:
                if ln.strip() == "}":
                    skip = 0
                continue
            out.append(ln)
        f.write_text("\n".join(out) + "\n")


def create_case(cfg: CaseConfig, tier: str, dest_root: Path, templates: Path) -> Path:
    """Create cases/<slug>/<tier> from the template; return the case path."""
    assert tier in ("RANS", "DDES"), tier
    dest_root = Path(dest_root)
    templates = Path(templates)
    case = dest_root / tier
    template = templates / f"case_{tier}"
    if case.exists():
        raise FileExistsError(f"Refusing to replace existing case: {case}")
    shutil.copytree(template, case)

    # 1. generated caseDefinition
    (case / "system" / "include" / "caseDefinition").write_text(case_definition(cfg, tier))

    # 2. geometry
    stl = Path(cfg.geometry["file"])
    if not stl.is_absolute():
        stl = templates.parent / stl
    tri = case / "constant" / "triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    (tri / "README.txt").unlink(missing_ok=True)
    if stl.exists():
        shutil.copy(stl, tri / "car.stl")
    else:
        print(f"  [warn] geometry not found: {stl} (place car.stl manually before meshing)")

    # 3. ground type
    if cfg.solve.get("ground_type", "moving") == "slip":
        _set_ground_slip(case)

    return case
