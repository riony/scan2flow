# Exploratory prototypes

Selected deterministic helper functions copied from existing local geometry/CFD scripts. Function bodies are unchanged; only import sets and module descriptions were supplied. They are not production modules and are not imported by the inspection CLI.

- `foam_dictionaries.py`: wrapping-oriented block/snappy/feature/decomposition dictionaries; no parameter validation or layer generation.
- `checkmesh_parser.py`: existing permissive parser, retained as a migration target. Its success heuristic is unsafe for an acceptance gate.
- `spline_fit.py`: chart-based robust fitting and bending penalty; no physical boundary recovery or validation.
- `cad_helpers.py`: spline pcurves and checked STEP read/write; needs a compatible optional OpenCASCADE runtime and integration tests.

Ground closure, full sewing and safe orchestration remain roadmap tasks. No archive paths, image loading, credentials or solver launchers were copied. The original source scripts remain unchanged. No claim is made that these helpers are already licensed for external distribution.
