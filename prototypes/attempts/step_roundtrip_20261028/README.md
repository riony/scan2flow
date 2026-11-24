# step_roundtrip_20261028

Historical date: 2026-10-28. Retrospective code extraction, not a new run.

Default large STEP transfer timed out. Compact planar FACETED_BREP serialization
and a single post-ReadFile orientation-repair override passed independent corner,
area, normal, volume and topology checks on the frozen mesh. An intentionally
inverted synthetic face was rejected. This verified isolated exchange only.

`compact_step.write_step(path, vertices, faces)` writes a new file using .17g
coordinates without merging or reorienting. Input must be a valid closed oriented
triangular mesh. STEP millimetres are an exchange convention, not a metric-scale
claim. `verify_facets` needs optional OCP (historically 8.0.1.1, 54 reader defaults),
NumPy/SciPy/trimesh and a caller-provided small default-reader STEP fixture.
`orientation_control.run(new_directory)` runs the optional synthetic negative check.

No archived geometry, stored correspondence/hash, native B-rep cache, original
large-file timing replay or default-import performance guarantee is supplied.
Assertions are validation guards; do not run verification with Python -O.

No captured images, camera data or saved source geometry are included.
Dependencies are optional; importing this package does not run a workflow.
Original numerical recipes and failed branches are retained. See provenance.json
for code hashes and extraction boundaries.
