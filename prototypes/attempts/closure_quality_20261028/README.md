# closure_quality_20261028

Historical date: 2026-10-28. Retrospective code extraction, not a new run.

The coarse wrap/remesh self-intersected. The fine wrap alone failed source-distance
and minimum-quality requirements. Later archived recovery, simplification, flips
and bounded optimization produced an isolated mesh pass; its CAD round trip was
still incomplete until the separate STEP attempt.

Two optional CGAL OFF command-line recipes: `wrap input.off prefix` and
`fine_wrap input.off prefix`. Each writes raw/output OFF and JSON. CGAL alpha-wrap,
remeshing and degeneracy-repair APIs plus their native dependencies are required.
All recipe numbers are preserved and expressed in the input coordinate units.

Only the two wrappers are imported. The later successful mesh-quality chain,
source samples, constraints, CAD export and acceptance evaluator are not reproduced.
Neither wrapper is a production repair or an automatic acceptance gate.

No captured images, camera data or saved source geometry are included.
Dependencies are optional; importing this package does not run a workflow.
Original numerical recipes and failed branches are retained. See provenance.json
for code hashes and extraction boundaries.
