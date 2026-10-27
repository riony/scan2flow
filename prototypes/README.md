# Deterministic migration sources

These are copies of working research scripts, with packaging and hygiene edits. They are **not a validated general-purpose pipeline**, and the installed CLI does not import them. The upstream geometry came from photogrammetry and learned multi-view depth, as described in the root README; inference is outside this package.

| Source group | Included work | Main migration limit |
| --- | --- | --- |
| `pipeline/config.py`, `casegen.py` | YAML configuration, turbulence quantities, reference area, case templates | Sample defaults and legacy schemas; explicit output/template arguments; existing cases refused |
| `pipeline/mesh.py`, `foam.py` | Orientation checks, meshing orchestration, quality and force parsing | Missing orientation metadata can bypass the gate; warning-only quality limits; permissive log success |
| `pipeline/ingest.py` | Repair, axis transform, scale, seating, symmetry cut, raster frontal area | Requires known length; no automatic nose-direction verification; half cut remains open |
| `pipeline/meshrepair.py`, `repair_prototype.py` | Surface wrap, voxel wrap, repair, damage, mirror/half-cut comparison | Wrapping may seal intended slots; resolution, memory and displacement need validation |
| `symmetry_frame.py`, `symmetry_closure.py` | Vehicle-frame estimation, bilateral augmentation, underbody closure, point-surface reconstruction | Historical region limits and symmetry assumptions remain; no accepted whole-body geometry copied |
| `tiered_assembly.py` | Regional height fields, prescribed solids, collars, sewing and topology checks | Historical assembly distorted a mirrored panel by about 36%; topology success did not validate its geometry |
| `sew_shell.py` | Full eleven-face shell reconciliation, sewing, named free edges, STEP and tessellation export | Region-specific schema, API-version assumptions and explicit scale evidence required |
| `curve_fit.py` | Plane initialization, training-only traces, bounded spline fits and interleaved cross-validation | Pre-annotated coordinates and cameras required; depth remains prior-sensitive; no image reads |
| `update_region_state.py` | Paired manifest/status updates with snapshots, hashes and recovery journal | Specific regional schema; recoverable replacements, not a filesystem-wide atomic transaction |

The original four helper extracts remain for comparison. `source_manifest.json` records source and copy hashes. `MIGRATION.md` lists the hygiene edits and the contracts that still need refactoring.

## Dependencies and execution

`pip install -e '.[prototypes]'` adds SciPy and PyYAML. Individual recipes also require compatible optional CAD/visualization libraries (OpenCASCADE bindings, VTK/PyVista, Open3D, OpenCV, Matplotlib or mesh boolean/voxel support). These heavy optional runtimes are deliberately not part of the inspection CLI. The clean-environment gate tests the CLI and solver-independent parsers, and syntax-checks every Python copy; it does not certify the optional CAD or solver recipes.

Use a disposable workspace containing copies of the required geometry, curve definitions and camera-coordinate data. Recipes retain legacy input filenames/schema; the two small mesh fixtures do not supply those inputs. Do not point these scripts at an evidence archive. Set `SCAN2FLOW_OF_BASHRC` explicitly before any optional solver call. No solver, rendering or reconstruction runs during install or the tests.

The shell wrappers still build command strings and accept trusted local inputs only; safe argument-vector orchestration remains milestone 8. `templates/case_RANS/Allclean` is destructive within its case. Full recipe execution can write outputs and invoke external software. The CLI is read-only.

## Log fixtures and milestone 7

`fixtures/checkMesh_sample.log` is a real **failed** mesh check, including excessive skewness. `fixtures/surfaceCheck_sample.log` is an **open** shell check, not a closed-surface success. Host, case, date and capture identifiers were stripped; numerical diagnostics were retained. Neither log belongs to the decimated test meshes. The surface log used a historical scale that was later withdrawn: its dimensional values are parser input only.

The copied `parse_checkmesh` still considers a truncated log successful if it lacks the word `Failed`. A regression test records that unsafe behavior explicitly so migration to a fail-closed parser can change it deliberately. Never use this prototype success field as a production acceptance decision.
