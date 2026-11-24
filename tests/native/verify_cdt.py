"""Small synthetic import-verification fixtures, not historical control reruns."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def call(binary, folder, name, points, edges):
    out = folder / name
    out.mkdir()
    text = f"{len(points)} {len(edges)}\n"
    text += "".join(f"{i} {x:.17g} {y:.17g}\n" for i, (x, y) in enumerate(points))
    text += "".join(f"{a} {b}\n" for a, b in edges)
    src = out / "input.txt"
    src.write_text(text)
    result = subprocess.run(
        [str(binary), str(src), str(out / "out")], capture_output=True, text=True, timeout=40
    )
    (out / "stdout.log").write_text(result.stdout)
    (out / "stderr.log").write_text(result.stderr)
    return out, result


def measure(points, faces):
    tri = np.asarray(points)[faces]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    area = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    assert count.max() <= 2
    return area, {tuple(e) for e in unique[count == 1]}, {tuple(e) for e in unique}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-binary", type=Path, required=True)
    ap.add_argument("--prior-binary", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for f in json.loads(args.fixtures.read_text()):
        points, loops = f["points"], f["loops"]
        segments = [
            (ring[i], ring[(i + 1) % len(ring)]) for ring in loops for i in range(len(ring))
        ]
        out, r = call(args.patch_binary, args.output, f["name"], points, segments)
        if "reject" in f:
            assert r.returncode != 0 and f["reject"] in r.stderr, (f["name"], r.stderr)
        else:
            assert r.returncode == 0, (f["name"], r.stderr)
            faces = np.loadtxt(out / "out.faces", dtype=int, skiprows=1).reshape(-1, 3)
            area, boundary, _ = measure(points, faces)
            assert boundary == {tuple(sorted(e)) for e in segments}
            assert set(faces.ravel()) == set(range(len(points)))
            assert abs(area - f["area"]) < 1e-12
        records.append({"fixture": f["name"], "passed": True, "adapter": "patch"})
    points = [(0, 0), (1, 0), (1, 1), (0, 1), (0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4)]
    out, r = call(args.prior_binary, args.output, "prior_donor_island", points, edges)
    assert r.returncode == 0, r.stderr
    faces = np.loadtxt(out / "out.faces", dtype=int, skiprows=1).reshape(-1, 3)
    area, boundary, all_edges = measure(points, faces)
    assert abs(area - 1.0) < 1e-12
    assert boundary == {tuple(sorted(e)) for e in edges[:4]}
    assert all(tuple(sorted(e)) in all_edges for e in edges)
    assert (4, 5, 6) in {tuple(sorted(f)) for f in faces}
    records.append({"fixture": "prior_donor_island", "passed": True, "adapter": "prior"})
    _, r = call(
        args.prior_binary, args.output, "prior_crossing", points[:4], edges[:4] + [(0, 2), (1, 3)]
    )
    assert r.returncode != 0 and "crossing constraints" in r.stderr
    records.append({"fixture": "prior_crossing", "passed": True, "adapter": "prior"})
    report = {
        "scope": "Synthetic import verification only; no historical control rerun",
        "passed": True,
        "fixtures": records,
    }
    (args.output / "assessment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
