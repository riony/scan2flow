# Geometry-only fixtures

These contain coordinates and triangle indices only: no images, textures, cameras, event metadata or source-archive sidecars.

| Fixture | Triangles | Preparation | Intended test |
|---|---:|---|---|
| open_shell.ply | 21,361 | Topology-preserving decimation of an existing sewn CAD tessellation, keeping boundary vertices | Open-shell inspection and future closure/sewing checks |
| rough_patch.ply | 37,846 | PCA-chart Delaunay triangles on noisy fused points; omit edges longer than four median nearest-neighbour spacings | Incomplete support, noise and boundary topology |

Both were translated independently to their bounding-box centres, with no scaling or filling. They are unrelated local coordinate frames and must not be assembled together. The rough triangulation is a test construction, not accepted physical surface geometry. Decimation can lose small shape details; neither fixture is CFD-ready or dimensional ground truth.

The input surfaces came from video photogrammetry and learned multi-view depth of a stationary vehicle. The shell also underwent smooth CAD fitting and sewing. All upstream inference, images and videos remain outside this repository. Source geometry was copied, never moved or changed. The disowned whole-car prototype is not used.

`manifest.json` records geometry-only hashes and transformation descriptions. The original detailed provenance stays with the private evidence archive. Redistribution rights have not been verified.

## Solver diagnostic samples

`checkMesh_sample.log` records a failed mesh check (excessive skewness); `surfaceCheck_sample.log` records an open shell. They are stripped historical logs, not fresh solver results and not checks of these decimated fixtures. Numeric content is retained solely for parser tests; the surface log's historical scale is not an accepted measurement. See `prototypes/README.md` for parser limitations.
