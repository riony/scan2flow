# hybrid_20261031

Historical date: 2026-10-31. Retrospective code extraction, not a new run.

Amendment 2's constrained closed-carrier hybrid coupon failed minimum triangle
quality: slivers already existed before panel transfer. Feature preservation did
not qualify the mesh. The downstream complete control did not run.

`remesh_carrier input.off feature_edges.txt edge_length output.off` requires a
closed triangular mesh and explicit vertex-index feature edges. It retains the
three-iteration constrained CGAL recipe. `transfer.transfer` takes caller-owned
carrier, donor and feature segments (optional stressed positions), returns a mesh
and processing provenance; NumPy/trimesh and optional VTK are required. Historical
fixed widths .013/.026 are in input coordinate units.

No clean-reference carrier, donor, stress geometry, quality evaluator or complete
control reproduction is provided. This is not disconnected-patch reconstruction;
closed topology and truthful feature support are caller prerequisites.

No captured images, camera data or saved source geometry are included.
Dependencies are optional; importing this package does not run a workflow.
Original numerical recipes and failed branches are retained. See provenance.json
for code hashes and extraction boundaries.
