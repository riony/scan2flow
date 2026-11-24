# Historical geometry attempts

These are retrospective imports made on 25 November 2026. Directory dates identify
the original work, not the Git commit time. The commits use their actual creation
dates. Each directory records original code hashes, the extracted functions and
the adaptations needed to accept caller-owned inputs.

The files are selected numerical kernels and optional tools, not complete replays
of the original experiments. No new captured geometry, camera observations,
depths, normals, images, masks, videos, weights or reconstruction workspaces are
included. Tiny coordinate fixtures are explicitly synthetic. The two pre-existing
geometry fixtures in the repository are unchanged.

| Original work | Prototype group | Outcome and reason for retaining it |
| --- | --- | --- |
| 16 Oct | [Robust spline fitting](spline_fitting_20261016/) | Spatial-median anchors and robust bicubic fits produced an open interior reference; missing-surface accuracy was not established. |
| 17 Oct | [Nonuniform spline](nonuniform_spline_20261017/) | Shoulder-focused knots retained; sampled bending later proved under-resolved under strip removal. |
| 26-28 Oct | [B-rep closure and named joins](cad_closure_20261028/) | Valid B-reps still collided; unoriented joins, UV retrim and partition variants failed or timed out. Kept as failed prototypes. |
| 28 Oct | [Exact return-mesh operations](return_mesh_20261028/) | Intersection splitting, exterior selection and topology repairs did not qualify the final mesh quality. |
| 28 Oct | [Coarse/fine closure wraps](closure_quality_20261028/) | Both isolated recipes needed further corrections; this subset does not reproduce the later complete quality-repair chain. |
| 28 Oct | [STEP round-trip correction](step_roundtrip_20261028/) | Compact faceted exchange and a guarded reader override passed isolated checks; this was not whole-body fidelity or CFD qualification. |
| 29 Oct | [Interfaces and transitions](interfaces_20261029/) | Individual trimming checks passed; full transition families still collided. No exterior union was accepted. |
| 31 Oct | [Dense and sparse wrap controls](wrap_controls_20261031/) | Cache/missing-feature defects were separated from later feature-retention failures. Original, saved-unrun and amended variants remain distinct. |
| 31 Oct | [Hybrid transfer coupon](hybrid_20261031/) | Locked-feature transfer retained geometry but carrier slivers failed quality. |
| 31 Oct | [Unconstrained retriangulation](retriangulation_20261031/) | The boundary guard rejected missing required segments. This remains the failed method, not a disguised constrained replacement. |
| 5 Nov | [Constrained triangulation](constrained_triangulation_20261105/) | Patch/seam proof and full saved carrier preserved locked segments; source kernels and synthetic fixtures retained. |
| 5 Nov | [Damaged-support closure prior](damaged_prior_20261105/) | Closed prior came from damaged support; raw-reader equality first failed and was corrected to check actual STL encoding. |
| 5 Nov | [Prior-remeshed hybrid](hybrid_qualification_20261105/) | Retained final evaluator passed geometry; original first-pass scores were unavailable and the construction runner failed during metadata assembly. |
| 5-6 Nov | [Matched CFD policy](matched_cfd_20261106/) | Initial mesh policy stopped the pair; calibrated policy and matched controls passed at one discretization. Pure diagnostic/gate kernels are retained, not an automatically released CFD pipeline. |
| 7 Nov | [Exact mesh sections](exact_sections_20261107/) | Kept every topological branch and demonstrated remaining departures; no branch cleanup or surface promotion. |
| 17 Nov | [Strip withholding and penalty audit](strip_withholding_20261117/) | Neither interpolant was promoted. The audit demonstrated under-integration; a corrected production fit was proposed but not run. |

Each group README states what was omitted, what the code requires and which
failure semantics remain. The directory index is a progress record, not a set of
product acceptance claims. Earlier variants are not recommended defaults.

## Scope exclusions

Camera calibration, learned matching, depth prediction, neural training and
normal-model comparisons stay outside this geometry-to-flow toolkit. The native
TSDF recovery on 6 November was an isolated runtime/version correction followed
by saved-depth/camera replay, so it has no geometry-only code import here. The
training-image coverage audit on 17 November likewise has no imported image,
camera or projection dataset. Their omission is a scope decision, not a claim
that the input surfaces were measured without those upstream methods.

## Verification and dependencies

Python synthetic tests exercise selected numerical kernels and known failure
classes. Native CDT fixtures have a separate [optional runner](../../tests/native/README.md).
CAD, VTK, OpenVDB and the remaining native recipes are optional migration sources;
passing the lightweight suite does not qualify them on an arbitrary mesh.
Dependencies retain their own licences. No native dependency code, compiled
libraries or executables are bundled.

The installed `scan2flow inspect` command does not import or dispatch these
prototypes. Use explicit disposable inputs and outputs, and bound expensive
operations externally. No scale, symmetry, physical fidelity or mesh readiness
is inferred from a successful prototype call.
