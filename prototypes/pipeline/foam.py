"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def of_bashrc() -> str:
    return os.environ["SCAN2FLOW_OF_BASHRC"]


def run(cmd: str, cwd: str | Path, log: str | None = None, check: bool = True) -> int:
    """Run a shell command with the OpenFOAM environment sourced."""
    cwd = Path(cwd)
    redirect = f" > {log} 2>&1" if log else ""
    full = f"source {of_bashrc()} >/dev/null 2>&1; cd {cwd!s}; {cmd}{redirect}"
    proc = subprocess.run(["bash", "-c", full])
    if check and proc.returncode != 0:
        tail = ""
        if log and (cwd / log).exists():
            tail = "\n".join((cwd / log).read_text(errors="ignore").splitlines()[-25:])
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\n{tail}")
    return proc.returncode


def foam_app(
    app: str,
    case: str | Path,
    parallel: bool = False,
    ncores: int = 1,
    log: str | None = None,
    args: str = "",
) -> int:
    """Run an OpenFOAM application, optionally under mpirun -parallel."""
    log = log or f"log.{app}"
    if parallel:
        cmd = f"mpirun -np {ncores} --oversubscribe {app} {args} -parallel"
    else:
        cmd = f"{app} {args}"
    return run(cmd, case, log=log)


# --------------------------------------------------------------------------- #
# Output parsers
# --------------------------------------------------------------------------- #
def _latest_coeff_file(case: Path, obj: str = "forceCoeffs1") -> Path | None:
    base = Path(case) / "postProcessing" / obj
    if not base.is_dir():
        return None
    candidates: list[Path] = []
    for tdir in base.iterdir():
        for name in ("coefficient.dat", "forceCoeffs.dat"):
            f = tdir / name
            if f.exists():
                candidates.append(f)
    # pick the smallest start-time directory (it accumulates the whole history)
    return min(candidates, key=lambda p: float(p.parent.name)) if candidates else None


_ALIAS = {  # normalise coefficient column names across OF versions
    "cd": "Cd",
    "cl": "Cl",
    "cm": "Cm",
    "cmpitch": "Cm",
    "cl(f)": "ClF",
    "clf": "ClF",
    "cl(r)": "ClR",
    "clr": "ClR",
    "cd(f)": "CdF",
    "cdf": "CdF",
    "cd(r)": "CdR",
    "cdr": "CdR",
}


def parse_force_coeffs(
    case: str | Path, obj: str = "forceCoeffs1", tail_frac: float = 0.25
) -> dict[str, float]:
    """Return time-averaged coefficients over the last ``tail_frac`` of the run.

    Robust to column-order differences between OF versions by reading the last
    ``#`` header line as the column map.
    """
    f = _latest_coeff_file(Path(case), obj)
    if f is None:
        return {}
    header: list[str] = []
    rows: list[list[float]] = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            toks = line.lstrip("#").split()
            if len(toks) > 2:  # a real column header, not a banner
                header = [t.lower() for t in toks]
            continue
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError:
            continue
    if not rows or not header:
        return {}
    n = max(1, int(len(rows) * tail_frac))
    tail = rows[-n:]
    means = [sum(col) / len(col) for col in zip(*tail)]
    out: dict[str, float] = {}
    for name, val in zip(header, means):
        key = _ALIAS.get(name)
        if key:
            out[key] = val
    # std of Cd over the window (unsteadiness indicator)
    if "time" in header:
        cd_idx = next((i for i, h in enumerate(header) if h == "cd"), None)
        if cd_idx is not None and len(tail) > 1:
            cds = [r[cd_idx] for r in tail]
            mu = sum(cds) / len(cds)
            out["Cd_std"] = (sum((c - mu) ** 2 for c in cds) / len(cds)) ** 0.5
    return out


def parse_checkmesh(case: str | Path, log: str = "log.checkMesh") -> dict[str, float]:
    """Extract the project quality gates from a checkMesh log."""
    p = Path(case) / log
    if not p.exists():
        return {}
    txt = p.read_text(errors="ignore")
    out: dict[str, float] = {}
    if m := re.search(r"non-orthogonality.*?Max:\s*([\d.eE+-]+)", txt):
        out["max_nonortho"] = float(m.group(1))
    if m := re.search(r"Max skewness\s*=\s*([\d.eE+-]+)", txt):
        out["max_skewness"] = float(m.group(1))
    if m := re.search(r"\bcells:\s*([\d]+)", txt):
        out["n_cells"] = float(m.group(1))
    out["ok"] = float("Mesh OK" in txt or "Failed" not in txt)
    return out
