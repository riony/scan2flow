"""v0.1 log parser. Known limitation: missing failure text can look successful. Do not use as an acceptance gate before fail-closed migration."""
import re
from pathlib import Path

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
