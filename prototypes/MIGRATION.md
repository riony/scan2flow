# Copy scope and hygiene changes

The deterministic source is included as migration material, not as evidence that the generic toolkit is complete. The source archive was read only. No capture frames, inspection images, model weights, scheduler configuration, remote credentials or machine-management scripts were copied.

Changes made during copying:

- Replaced module descriptions with explicit prototype scope. Removed project names, dated folder names and private absolute paths; preserved the root README's upstream provenance.
- Kept complete pipeline modules and recipe calculations. Removed the curve fitter's capture-image overlay block; numerical projection, ray initialization, spline fitting, model selection and chart plots remain.
- Moved the curve fitter's top-level execution under `main(source, output)`. Its serialized camera model is trusted local input only; the legacy pickle loader must not consume untrusted files.
- Made case/template and ingestion output paths arguments. Replaced automatic deletion of existing case/wrap workspaces with refusal. Removed repair-script import-time directory creation and argument parsing.
- Made legacy symmetry/closure/assembly recipe roots explicit `--workspace` and `--artifact-root` inputs. The workspace still holds both recipe inputs and generated outputs, and some scratch files are removed there. It must be a disposable copy.
- Replaced the fixed solver installation path with `SCAN2FLOW_OF_BASHRC`. Shell quoting and return-code handling otherwise retain prototype limitations.
- Replaced historical default metric factors with required `--mm-per-unit` values for assembly recipes. Sewing requires `--upper-dir`, `--artifact-root`, `--out-dir` and `--scale-json`. Its `recommended_metric_scale` object requires `primary_mm_per_unit`, `sensitivity_upper_mm_per_unit`, `midpoint_mm_per_unit`, `relative_statistical_uncertainty_pct` and `relative_sensitivity_pct`. These are caller assertions, not verified measurements; no master geometry has been scaled here.
- Retained historical region constants, smoothing, prescribed dimensions and incomplete quality heuristics. The old 2000 mm tyre-inclusive-width comment was corrected; the prescription itself remains experimental. Reusing the rejected assembly output as accepted geometry is explicitly out of scope.
- Stripped host/path identifiers from log samples; preserved tool versions, warnings and numerical values. Copied the complete RANS template. No DDES template is included, even though the legacy configuration can name that tier.
- Applied formatting, import cleanup and local lint settings. Prototype-only unused intermediates and lambda spelling are exempt; undefined names and syntax remain checked. Core code and tests use the full configured correctness checks.

Before migrating a recipe into `src/scan2flow`, give it a documented generic input contract, safe output handling and geometric tests. CAD binding compatibility, end-to-end solver execution, source-scale correctness, physical fidelity and closed-car production are not certified by a passing inspection test suite.

On 28 October 2026, the project namespace and configuration prefix were renamed to `scan2flow` and `SCAN2FLOW`. This changes imports, console commands, diagnostic branding and the optional solver environment variable; numerical calculations and fixture bytes are unchanged. The source manifest retains the original copy hashes.
