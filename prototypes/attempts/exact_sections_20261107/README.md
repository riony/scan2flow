# All-branch mesh section diagnostic

Historical date: 7 November 2026. Imported as a prototype on 25 November 2026.

The original diagnostic intersected complete saved triangle meshes with fixed
planes. It retained every branch and selected no replacement surface. The cuts
showed that prior section departures were present in literal mesh intersections,
while physical accuracy and branch visibility remained unresolved.

`sections.py` accepts generic vertices, triangles and a plane. Connections use
source vertex and edge identity, so coincident disconnected topology stays
separate. Every segment retains triangle ownership and belongs to one branch.
Point tangencies, coplanar facets, zero-length cuts and junctions are reported
explicitly. Independent Trimesh intersections check transverse triangles.
The module includes the original analytic fixtures, without real mesh inputs,
source protocols, plotting or export orchestration.

"Exact" means analytic intersections of the supplied piecewise-linear mesh in
float64. It does not mean exact-predicate arithmetic, certification of
self-intersections or physical accuracy. The absolute epsilon keeps its original
input-unit meaning; this prototype is not a production acceptance gate.

Synthetic regression covers coincident boxes, branch and segment preservation,
special cases and an empty intersection. No captured mesh was loaded or rescored.
`provenance.json` records original source hashes, symbol ranges and adaptations.
