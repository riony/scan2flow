"""Deterministic geometry/CFD prototype copied from a working research pipeline.
Inputs and legacy schemas are documented in README.md and MIGRATION.md.
Paths and execution boundaries were adapted; numerical routines remain exploratory.
No capture images or upstream inference are included or required.
Historical thresholds, dimensions and quality heuristics are not generic guarantees.
Do not run against an evidence archive; use disposable copies and explicit outputs.
"""

import copy
import hashlib
import json
import os
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def synchronize(manifest, acceptance):
    result = copy.deepcopy(acceptance)
    result["updated"] = manifest["updated"]
    result["authoritative_current_report"] = manifest["source_review"]
    result["plan_of_record"] = manifest["plan_of_record"]
    result["workflow_policy"] = copy.deepcopy(manifest["workflow_policy"])
    result["mandatory_camera_policy"] = manifest["known_global_model_defects"][0][
        "mandatory_mitigation"
    ]
    # The original defect remains historical evidence; avoid an obsolete policy tail.
    defect = manifest["known_global_model_defects"][0]
    result["known_global_model_defect"] = defect["script"] + ": " + defect["defect"]
    next_region = manifest["next_region"]
    # Carry the active regional scope forward instead of retaining the legacy
    # nose/chassis label after a new region has been explicitly authorized.
    result["whole_car_expansion"] = "paused_beyond_authorized_" + next_region["id"]
    candidate = result["regional_candidate"]
    if candidate["id"] != next_region["id"]:
        # Starting a new budgeted region must retain the previous disposition.
        result.setdefault("regional_candidate_history", {})[candidate["id"]] = copy.deepcopy(
            candidate
        )
        candidate = {"id": next_region["id"]}
        result["regional_candidate"] = candidate
    candidate["status"] = next_region["execution_status"]
    candidate["run_budget"] = copy.deepcopy(next_region["run_budget"])
    candidate["camera_policy"] = next_region["camera_policy"]
    if "aft_subregion" not in candidate:
        region = next(r for r in manifest["regions"] if r["id"] == candidate["id"])
        candidate["region"] = copy.deepcopy(region)
        candidate["geometry_scaled"] = manifest["scale"]["applied_to_master_geometry"]
        candidate["remaining_gates"] = copy.deepcopy(next_region.get("remaining_gates", []))
        result["registration_status"] = region["registration_status"]
        result["regional_local_ba_model"] = region.get("camera_model")
        result["evidence"] = next_region["current_evidence"]
        return result
    aft = next(r for r in manifest["regions"] if r["id"] == candidate["aft_subregion"]["id"])
    target = candidate["aft_subregion"]
    for key in [
        "status",
        "source",
        "source_status",
        "selected_pass",
        "accepted_for_cad",
        "single_pass_reference",
    ]:
        target[key] = copy.deepcopy(aft[key])
    # Carry final export decisions and current gates from the same authority.
    # Optional keys preserve compatibility with older manifests.
    for key in ["export_artifacts", "validation", "budget_disposition", "reference_cloud"]:
        if key in aft:
            target[key] = copy.deepcopy(aft[key])
    if "remaining_gates" in next_region:
        candidate["remaining_gates"] = copy.deepcopy(next_region["remaining_gates"])
    result["registration_status"] = aft["aft_registration_status"]
    return result


def commit(root, proposed_manifest, audit_dir):
    root, audit_dir = Path(root), Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=False)
    names = ["region_manifest.json", "ACCEPTANCE_STATUS.json"]
    before = {name: (root / name).read_bytes() for name in names}
    acceptance = synchronize(proposed_manifest, json.loads(before[names[1]]))
    after = dict(zip(names, [encode(proposed_manifest), encode(acceptance)]))
    assert proposed_manifest["source_review"] == acceptance["authoritative_current_report"]
    for name in names:
        (audit_dir / (name + ".before")).write_bytes(before[name])
        (audit_dir / (name + ".after")).write_bytes(after[name])
        # Stage next to target so replacement stays on the same filesystem.
        path = root / ("." + name + ".pending")
        with path.open("xb") as f:
            f.write(after[name])
            f.flush()
            os.fsync(f.fileno())
    journal = {
        "status": "prepared",
        "before": {n: digest(v) for n, v in before.items()},
        "after": {n: digest(v) for n, v in after.items()},
        "scope": "Metadata only; no reconstruction, geometry, camera or validation operation.",
    }
    jp = audit_dir / "transaction.json"
    jp.write_bytes(encode(journal))
    assert all((root / n).read_bytes() == before[n] for n in names), (
        "Concurrent state edit; inspect prepared transaction before retry."
    )
    for name in names:
        os.replace(root / ("." + name + ".pending"), root / name)
    assert all((root / n).read_bytes() == after[n] for n in names)
    journal["status"] = "committed"
    jp.write_bytes(encode(journal))
    return acceptance
