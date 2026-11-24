# retriangulation_20261031

Historical date: 2026-10-31. Retrospective code extraction, not a new run.

Amendment 3's planar Delaunay coupon aborted at the boundary-preservation guard
before writing a replacement carrier. This was an implementation failure;
damaged-input and CFD work did not run. The preserved algorithm is unconstrained
Delaunay plus rejection checks, not a constrained triangulator.

`patch(vertices, faces, max_rounds=12)` and `repair(mesh, locked_edges)` accept
caller-provided arrays/trimesh meshes. NumPy/SciPy/trimesh are required. Coordinates,
planarity threshold 1e-7, minimum quality .05, and removal rules are unchanged.
Functions return proposed geometry/audit or raise when prerequisites fail.

No saved carrier/donor arrays, actual failed coupon, subsequent triangulation
method or qualification success is reproduced. Assertions and exception guards
must remain enabled; the source is retained unchanged apart from formatting.

No captured images, camera data or saved source geometry are included.
Dependencies are optional; importing this package does not run a workflow.
Original numerical recipes and failed branches are retained. See provenance.json
for code hashes and extraction boundaries.
