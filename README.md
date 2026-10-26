# scan2foam

An experimental computational-geometry and OpenFOAM pre-processing toolkit for surface scan meshes. Version 0.1 provides **read-only mesh inspection**. Automatic repair, verified closure, case generation and CFD-quality gates are roadmap work; this scaffold does not yet turn a scan into a CFD-ready car.

## Install and inspect

Python 3.11 or later:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
scan2foam inspect fixtures/open_shell.ply
scan2foam inspect fixtures/rough_patch.ply
pytest -q
```

The command emits JSON with face/vertex counts, bounds, edge incidence, boundary loops/chains/branches, Euler number and watertightness. It does not alter the input or infer units. Coincident coordinates are merged to 12 decimal places for topology analysis; no self-intersection or vertex-manifold certificate is implied. A watertight flag is not aerodynamic or CFD validation.

## Inputs and provenance

The fixtures are geometry-only derivatives of stationary-vehicle surface reconstruction. The upstream sources used video photogrammetry and learned multi-view depth; one fixture also underwent smooth CAD fitting and sewing. Upstream reconstruction software, weights, video, photographs and inspection images are outside this repository's scope. They are not required to run the deterministic toolkit. See `fixtures/README.md` for transformations and limitations.

No footage, textures or bystander imagery is included. The numeric geometry remains unscaled and must not be assumed to be metres or millimetres. Fixture suitability as a test input does not establish permission for external redistribution.

## Layout and current limits

- `src/scan2foam/`: inspection CLI and empty geometry/OpenFOAM namespaces.
- `prototypes/`: copied exploratory helper functions; migration and validation remain unfinished.
- `fixtures/`: one decimated open shell and one noisy support patch.
- `tests/`: CLI/read-only and basic topology checks.
- `ROADMAP.md`: bounded engineering milestones, including provenance and acceptance gates.

This is a separate repository built from copies. Its setup does not move or rewrite the source evidence archive. Prototype functions are not automatically executed by the CLI. No solver is launched during installation or tests.

## Release status

Local scaffold only: no remote, upload, recording or publication has been configured. No distribution licence has been selected. Review ownership, employment IP obligations, fixture redistribution rights and any recording/transcript agreement before assigning a licence or sharing it. No claim is made about any platform's eligibility rules, pay rate or contract terms.
