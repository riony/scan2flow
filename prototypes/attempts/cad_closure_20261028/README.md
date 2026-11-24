# cad_closure_20261028

Historical date: 2026-10-26 protocol; construction/corrections 2026-10-28. Retrospective code extraction, not a new run.

Initial reflected boundary bridges made a valid B-rep that self-intersected.
Centreline clipping and orientation-corrected named joins still failed collisions.
The unoriented variant, UV-chord retrim (invalid B-rep) and volume partition
(timeout) remain failed diagnostics. No whole assembly or CFD acceptance occurred.

`closures.py` accepts OCP shapes in a caller-defined frame, symmetric about y=0.
Caller supplies clip bounds; functions return new shapes and limited topology
metadata. `retrim_uv_chord.retrim` takes explicit edge/face pairs. NumPy and
optional OCP are required. Bound expensive CAD calls externally.

Saved transforms, boundary/name matching, source geometry, full-car prescriptions,
tessellation exports, collision tests and original execution budgets are omitted.
The pure helpers do not assert physical symmetry or nonintersection.

No captured images, camera data or saved source geometry are included.
Dependencies are optional; importing this package does not run a workflow.
Original numerical recipes and failed branches are retained. See provenance.json
for code hashes and extraction boundaries.
