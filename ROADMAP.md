# Roadmap

Each item is a bounded implementation slice with a concrete check, not a promise that all geometry cases fit a single session.

1. **Provenance and acceptance schema:** separate observed/interpolated/prescribed/mirrored origins from coarse acceptance and assembly readiness; preserve hashes and units, reject missing required metadata.
2. **Boundary topology and repair:** distinguish loops, branches and non-manifold vertices; remove degenerates without silently closing aero slots. Test known topology and geometry displacement.
3. **Ground plane and prescribed closures:** establish a user-supported datum, clip/cap with explicit assumptions, test manifoldness and preserve observed geometry. Do not fit a ground plane to the sloped body by default.
4. **Aerodynamic frame and symmetry:** explicit flow axis, centreline and ground clearance; mirror only with provenance and opposite-side checks. Export named surface regions without regional rescaling.
5. **CAD fitting, trimming and sewing:** migrate spline and STEP helpers; join compatible boundaries, round-trip STEP and tessellate. Test seams and topology separately from shape fidelity.
6. **Wind-tunnel and refinement dictionaries:** migrate prototype blockMesh and snappy writers, parameterize domain/blockage and refinement/layers, validate dimensions and locationInMesh.
7. **OpenFOAM quality gates:** migrate surfaceCheck/checkMesh parsing with fail-closed missing/unknown results; distinguish tool failure from passing geometry and retain complete diagnostics.
8. **Case orchestration:** safe, explicit output directories and subprocess arguments, no implicit deletion; one small solver-independent dictionary test, then an optional local meshing integration case.

The prototypes remain exploratory. No closed whole-car fixture or automatic pass is assumed. Changes to inference, camera calibration or depth methods are outside this package's scope.
