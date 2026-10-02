"""Integration checks on the real V2 registrations (the seams the V1 unit tests missed; R2 Section 5.2).

Every real cell must parse through the candidate schema the bank builder uses,
load its registered item lists in registered order with the ordered hash, keep
evaluation items out of the preparation stage, and (021C) equal the prefix of
its 021B parent.  The planner must produce the six-GPU sampling and preparation
plans from the real registrations, and the approval template must authorize nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import candidates_ext, guard_ext, items_ext, plan_ext  # noqa: E402
from satml2027ext._common import canonical_json_hash, read_json, write_json_atomic  # noqa: E402

CONFIGS = PROJECT / "configs" / "satml2027ext"
ITEMS = PROJECT / "results" / "satml2027ext" / "items"
PATHS = {experiment_id: CONFIGS / f"{experiment_id.lower()}.json" for experiment_id in guard_ext.EXPERIMENT_IDS}
needs_registrations = pytest.mark.skipif(not all(path.is_file() for path in PATHS.values()), reason="V2 registrations not generated")


def registrations() -> dict[str, dict]:
    return {experiment_id: guard_ext.load_registration(path) for experiment_id, path in PATHS.items()}


@needs_registrations
def test_every_real_cell_parses_and_loads_its_lists_in_registered_order():
    loaded = registrations()
    total = 0
    for experiment_id, registration in loaded.items():
        subset = candidates_ext.is_subset_registration(registration)
        control_role = registration["bank_construction"].get("control_training", {}).get("role")
        for cell in registration["cells"]:
            ids = candidates_ext.registered_candidate_ids(registration, cell)
            assert len(ids) == (3 if subset else 36), cell["cell_id"]
            rows = items_ext.load_items(ITEMS, cell, "evaluation", registration)
            items_ext.verify_item_list(rows, cell, "evaluation")
            with pytest.raises(RuntimeError):  # evaluation items never reach the preparation stage
                items_ext.verify_preparation_sources(rows, "development")
            if not subset:
                for role in {"development", control_role}:
                    role_rows = items_ext.load_items(ITEMS, cell, role, registration)
                    items_ext.verify_item_list(role_rows, cell, role)
                    items_ext.verify_preparation_sources(role_rows, role)
            total += 1
    assert total == 36


@needs_registrations
def test_021c_is_the_prefix_of_021b_and_names_its_parent():
    loaded = registrations()
    b, c = loaded["EXP-20260921-021B"], loaded["EXP-20260921-021C"]
    assert candidates_ext.subset_parent(c)["registration_sha256"] == b["registration_sha256"]
    cells_b = {cell["cell_id"]: cell for cell in b["cells"]}
    for cell in c["cells"]:
        parent = cells_b[cell["parent_cell_id"]]
        child_rows = items_ext.load_items(ITEMS, cell, "evaluation", c)
        parent_rows = items_ext.load_items(ITEMS, parent, "evaluation", b)
        assert [row["item_id"] for row in child_rows] == [row["item_id"] for row in parent_rows[:100]]
        assert cell["model_id"] == parent["model_id"] and cell["sigma"] == parent["sigma"] == 0.25


@needs_registrations
def test_registrations_share_data_models_and_the_item_manifest():
    loaded = registrations()
    blocks = [canonical_json_hash(registration["data_roots"]) for registration in loaded.values()]
    assert len(set(blocks)) == 1
    manifest_hash = canonical_json_hash(read_json(ITEMS / "item_manifest_ext.json"))
    assert {registration["provenance"]["item_manifest_sha256"] for registration in loaded.values()} == {manifest_hash}
    for registration in loaded.values():
        for cell in registration["cells"]:
            assert registration["models"][cell["model_id"]]["sha256"]
    assert loaded["EXP-20260921-021A"]["sealed_evaluation_access"] is False
    assert all(not cell["consumes_sealed_split"] for cell in loaded["EXP-20260921-021A"]["cells"])


@needs_registrations
def test_the_approval_template_binds_everything_and_authorizes_nothing(tmp_path):
    loaded = registrations()
    template = guard_ext.approval_template({key: value["registration_sha256"] for key, value in loaded.items()})
    closure = template["source_sha256"]
    for path in PATHS.values():
        assert path.relative_to(PROJECT).as_posix() in closure
    assert "satml2027ext/prepare_banks_ext.py" in closure and "satml2027/day1/d1_03b_train_controls.py" in closure
    write_json_atomic(tmp_path / "approval.json", template)
    for stage in guard_ext.STAGES:
        with pytest.raises(guard_ext.ApprovalError):
            guard_ext.verify_approval(loaded["EXP-20260921-021B"], stage=stage, approval_path=tmp_path / "approval.json")
    assert not guard_ext.DEFAULT_APPROVAL_PATH.exists()


@needs_registrations
def test_six_gpu_sampling_and_preparation_plans_from_the_real_registrations(tmp_path):
    paths = [PATHS[key] for key in guard_ext.EXPERIMENT_IDS]
    assert plan_ext.main(["--registrations", *map(str, paths), "--devices", "5090:6", "--out", str(tmp_path / "sampling")]) == 0
    plan = read_json(tmp_path / "sampling" / "plan.json")
    assert plan["summary"]["cell_count"] == 36
    assert all(job["item_count"] <= 1000 for job in plan["jobs"])
    assert plan["sealed_evaluation_access"] is True
    text = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "sampling" / "launch").glob("run_gpu*.sh"))
    assert text.count("--parent-registration") == sum(job["experiment_id"] == "EXP-20260921-021C" for job in plan["jobs"])
    assert plan_ext.main(["--registrations", *map(str, paths), "--devices", "5090:6", "--out", str(tmp_path / "prep"),
                          "--stage", "preparation"]) == 0
    scripts = list((tmp_path / "prep" / "launch_preparation").glob("prepare_gpu*.sh"))
    assert sum(path.read_text(encoding="utf-8").count("prepare_banks_ext.py") for path in scripts) == 30
