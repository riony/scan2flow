# scan2flow

An experimental computational-geometry and OpenFOAM pre-processing toolkit for surface scan meshes. Version 0.1 provides **read-only mesh inspection**. Automatic repair, verified closure, case generation and CFD-quality gates are roadmap work; this scaffold does not yet turn a scan into a CFD-ready car.

## Install and inspect

Python 3.11 or later:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
scan2flow inspect fixtures/open_shell.ply
scan2flow inspect fixtures/rough_patch.ply
pytest -q
ruff check .
```

The command emits JSON with face/vertex counts, bounds, edge incidence, boundary loops/chains/branches, Euler number and watertightness. It does not alter the input or infer units. Coincident coordinates are merged to 12 decimal places for topology analysis; no self-intersection or vertex-manifold certificate is implied. A watertight flag is not aerodynamic or CFD validation.

## Inputs and provenance

The fixtures are geometry-only derivatives of stationary-vehicle surface reconstruction. The upstream sources used video photogrammetry and learned multi-view depth; one fixture also underwent smooth CAD fitting and sewing. Upstream reconstruction software, weights, video, photographs and inspection images are outside this repository's scope. They are not required to run the deterministic toolkit. See `fixtures/README.md` for transformations and limitations. The project was developed on personal equipment using personal footage.

No footage, textures or bystander imagery is included. The numeric geometry remains unscaled and must not be assumed to be metres or millimetres.

## Layout and current limits

- `src/scan2flow/`: inspection CLI and empty geometry/OpenFOAM namespaces.
- `prototypes/`: complete deterministic pipeline prototypes and research recipes; migration and validation remain unfinished.
- `fixtures/`: one decimated open shell and one noisy support patch.
- `tests/`: CLI/read-only and basic topology checks.
- `ROADMAP.md`: bounded engineering milestones, including provenance and acceptance gates.

This is a separate repository built from copies. Its setup does not move or rewrite the source evidence archive. Prototype functions are not automatically executed by the CLI. No solver is launched during installation or tests.

## Licence and verification

The code and included geometry fixtures are provided under the [MIT licence](LICENSE), copyright 2026 riony. Dependencies retain their own licences. The owner has confirmed employment-IP clearance.

The project was renamed from `scan2foam` to `scan2flow` on 28 October 2026. Use the `scan2flow` CLI, `scan2flow` Python imports and `SCAN2FLOW_OF_BASHRC` for optional solver prototypes. Existing installations should be reinstalled with the commands above; no legacy command or environment-variable alias is provided.

See `verification.json` for the original pre-rename installation, tests, lint and repository audit, and `rename_verification.json` for the rename checks. The original record retains the former package name and publication state as historical evidence. These checks certify the inspection package and source hygiene, not the optional solver/CAD recipes or physical geometry. Publication and recorded sessions are separate actions; no platform eligibility, pay rate or contract terms are asserted here.
