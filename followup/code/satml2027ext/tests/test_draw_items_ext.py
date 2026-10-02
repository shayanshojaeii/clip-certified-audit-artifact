from __future__ import annotations

import csv
import filecmp
import hashlib
import json
from pathlib import Path

import pytest

from conftest import CREATED_AT, EXP019, PHASE3, PHASE4, draw_args, require_real_inputs

import draw_items_ext as di
from _common import canonical_json_hash, sha256_id_list

A, B, C = di.EXPERIMENT_IDS["021a"], di.EXPERIMENT_IDS["021b"], di.EXPERIMENT_IDS["021c"]


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row["item_id"] for row in csv.DictReader(handle)]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def phase3_rows() -> list[dict[str, str]]:
    return di.load_rows(PHASE3)


# ---------------------------------------------------------------- ranking


def test_rank_key_is_salted_sha256():
    salt = di.draw_salt(B)
    assert salt == "EXP-20260921-021B:item_draw:v1"
    expected = hashlib.sha256(f"{salt}:cifar100:cifar100-test-1".encode()).hexdigest()
    assert di.rank_key(salt, "cifar100", "cifar100-test-1") == expected
    assert di.rank_key(di.draw_salt(A), "cifar100", "cifar100-test-1") != expected


def test_stratified_draw_round_robin_and_exclusion():
    pool = [{"dataset_id": "d", "item_id": f"i{k}", "dataset_index": str(k), "label": str(k % 3), "source_split": "s"} for k in range(30)]
    drawn = di.stratified_draw(pool, 2, set(), "salt", class_count=3)
    assert [row["label"] for row in drawn] == ["0", "1", "2", "0", "1", "2"]
    ranked0 = sorted((row for row in pool if row["label"] == "0"), key=lambda row: di.rank_key("salt", "d", row["item_id"]))
    assert drawn[0]["item_id"] == ranked0[0]["item_id"] and drawn[3]["item_id"] == ranked0[1]["item_id"]
    again = di.stratified_draw(pool, 2, {ranked0[0]["item_id"]}, "salt", class_count=3)
    assert again[0]["item_id"] == ranked0[1]["item_id"]
    with pytest.raises(RuntimeError, match="only"):
        di.stratified_draw(pool, 11, set(), "salt", class_count=3)
    with pytest.raises(RuntimeError, match="classes"):
        di.stratified_draw(pool, 1, set(), "salt", class_count=4)


# ---------------------------------------------------------------- full build (synthetic pixels)


def test_counts_full_build(full_build):
    out, manifest = full_build
    assert manifest["synthetic_inputs_allowed"] is True and manifest["pending"] == []
    a = manifest["studies"][A]["datasets"]
    for dataset, per_fold in (("cifar100", 500), ("eurosat", 50)):
        for fold in ("0", "1", "2"):
            for role in ("evaluation", "development"):
                info = a[dataset]["folds"][fold][role]
                assert info["status"] == "written" and info["count"] == per_fold
                assert len(read_ids(out / info["csv"])) == per_fold
    b = manifest["studies"][B]["datasets"]
    expected = {
        "cifar100_test": {"evaluation": 3000, "development": 1000, "control_train": 2000},
        "eurosat_sealed": {"evaluation": 1000, "development": 300, "control_train": 200},
        "imagenette": {"evaluation": 3000, "development": 300, "control_train": 200},
    }
    for dataset, roles in expected.items():
        for role, count in roles.items():
            info = b[dataset]["splits"][role]
            assert info["status"] == "written" and info["count"] == count, (dataset, role)
            ids = read_ids(out / info["csv"])
            assert len(ids) == count and sha256_id_list(ids) == info["item_ids_sha256"]
    c = manifest["studies"][C]["datasets"]
    for dataset in expected:
        assert c[dataset]["status"] == "written" and c[dataset]["count"] == 100
        parent = read_ids(out / b[dataset]["splits"]["evaluation"]["csv"])
        assert read_ids(out / c[dataset]["csv"]) == parent[:100]
        assert c[dataset]["parent_item_ids_sha256"] == b[dataset]["splits"]["evaluation"]["item_ids_sha256"]


def test_class_balance_of_prefixes(full_build):
    out, manifest = full_build
    b = manifest["studies"][B]["datasets"]
    for dataset, class_count in (("cifar100_test", 100), ("eurosat_sealed", 10), ("imagenette", 10)):
        for role in ("evaluation", "development", "control_train"):
            rows = read_rows(out / b[dataset]["splits"][role]["csv"])
            labels = [int(row["label"]) for row in rows]
            assert labels[:class_count] == list(range(class_count)), (dataset, role)
            per_class = {label: labels.count(label) for label in set(labels)}
            assert len(per_class) == class_count and len(set(per_class.values())) == 1
    subset = read_rows(out / manifest["studies"][C]["datasets"]["eurosat_sealed"]["csv"])
    assert sorted({row["label"] for row in subset}) == [str(k) for k in range(10)]
    assert all([row["label"] for row in subset].count(str(k)) == 10 for k in range(10))


def test_disjointness_across_roles_and_sources(full_build):
    out, manifest = full_build
    b = manifest["studies"][B]["datasets"]
    for dataset in b:
        seen: set[str] = set()
        for role in ("evaluation", "development", "control_train"):
            ids = set(read_ids(out / b[dataset]["splits"][role]["csv"]))
            assert not (ids & seen), (dataset, role)
            seen |= ids
    # evaluation sources are what the plan says
    cifar_eval = read_rows(out / b["cifar100_test"]["splits"]["evaluation"]["csv"])
    assert all(row["item_id"].startswith("cifar100-test-") and row["source_split"] == "cifar100_official_test" for row in cifar_eval)
    cifar_dev = read_rows(out / b["cifar100_test"]["splits"]["development"]["csv"])
    assert all(row["item_id"].startswith("cifar100-train-") for row in cifar_dev)
    im_eval = read_rows(out / b["imagenette"]["splits"]["evaluation"]["csv"])
    im_dev = read_rows(out / b["imagenette"]["splits"]["development"]["csv"])
    assert all("/val/" in row["item_id"] for row in im_eval) and all("/train/" in row["item_id"] for row in im_dev)


def test_synthetic_eval_matches_labels(full_build, synthetic_root):
    out, manifest = full_build
    labels = di.cifar100_test_labels(synthetic_root, require_official=False)
    rows = read_rows(out / manifest["studies"][B]["datasets"]["cifar100_test"]["splits"]["evaluation"]["csv"])
    assert all(labels[int(row["dataset_index"])] == int(row["label"]) for row in rows)


# ---------------------------------------------------------------- real-only build (no pixels)


def test_real_only_build_pending_and_written(real_only_build):
    out, manifest = real_only_build
    assert manifest["synthetic_inputs_allowed"] is False
    assert manifest["final_test_access"]["cifar100_official_test_labels_read"] is False
    b = manifest["studies"][B]["datasets"]
    assert b["cifar100_test"]["splits"]["evaluation"]["status"] == "pending"
    assert b["cifar100_test"]["splits"]["evaluation"]["item_ids_sha256"] == di.PENDING
    assert b["cifar100_test"]["splits"]["development"]["status"] == "written"
    assert b["cifar100_test"]["splits"]["control_train"]["status"] == "written"
    assert all(b["eurosat_sealed"]["splits"][role]["status"] == "written" for role in ("evaluation", "development", "control_train"))
    assert all(b["imagenette"]["splits"][role]["status"] == "pending" for role in ("evaluation", "development", "control_train"))
    c = manifest["studies"][C]["datasets"]
    assert c["eurosat_sealed"]["status"] == "written" and c["cifar100_test"]["status"] == "pending" and c["imagenette"]["status"] == "pending"
    assert len(manifest["pending"]) == 6
    assert all("Users" not in entry["reason"] and ":\\" not in entry["reason"] for entry in manifest["pending"])
    assert manifest["inputs"]["phase3_assignments"]["path"] == "artifacts/day14/phase3_splits_v1/assignments.csv"


def test_lists_shared_between_builds_are_identical(full_build, real_only_build):
    out_full, m_full = full_build
    out_real, m_real = real_only_build
    a_full, a_real = m_full["studies"][A]["datasets"], m_real["studies"][A]["datasets"]
    for dataset in a_full:
        for fold in ("0", "1", "2"):
            for role in ("evaluation", "development"):
                assert a_full[dataset]["folds"][fold][role]["item_ids_sha256"] == a_real[dataset]["folds"][fold][role]["item_ids_sha256"]
    b_full, b_real = m_full["studies"][B]["datasets"], m_real["studies"][B]["datasets"]
    for dataset, roles in (("cifar100_test", ("development", "control_train")), ("eurosat_sealed", ("evaluation", "development", "control_train"))):
        for role in roles:
            assert b_full[dataset]["splits"][role]["item_ids_sha256"] == b_real[dataset]["splits"][role]["item_ids_sha256"]
            assert filecmp.cmp(out_full / f"exp021b__{dataset}__{role}.csv", out_real / f"exp021b__{dataset}__{role}.csv", shallow=False)


def test_exp021a_lists_equal_phase3_folds(real_only_build):
    out, manifest = real_only_build
    rows = phase3_rows()
    for dataset in ("cifar100", "eurosat"):
        for fold in ("0", "1", "2"):
            for role, phase3_role in (("evaluation", "calibration_validation"), ("development", "calibration_development")):
                expected = [row["item_id"] for row in rows if row["dataset_id"] == dataset and row["phase3_role"] == phase3_role and row["fold"] == fold]
                got = read_ids(out / f"exp021a__{dataset}__fold{fold}__{role}.csv")
                assert got == expected
            assert not set(read_ids(out / f"exp021a__{dataset}__fold{fold}__evaluation.csv")) & set(read_ids(out / f"exp021a__{dataset}__fold{fold}__development.csv"))


def test_sealed_list_is_exactly_the_phase3_sealed_ids(real_only_build):
    out, manifest = real_only_build
    sealed = {row["item_id"] for row in phase3_rows() if row["dataset_id"] == "eurosat" and row["phase3_role"] == "final_test_sealed"}
    got = read_ids(out / "exp021b__eurosat_sealed__evaluation.csv")
    assert set(got) == sealed and len(got) == 1000 and len(set(got)) == 1000
    assert manifest["final_test_access"]["pixels_loaded"] is False


def test_reserve_draws_exclude_consumed_roles_phase4_and_exp019(real_only_build):
    out, manifest = real_only_build
    rows = phase3_rows()
    role_by_id = {(row["dataset_id"], row["item_id"]): row["phase3_role"] for row in rows}
    explicit = {row["item_id"] for row in rows if row["phase3_role"] == "excluded_explicit"}
    assert len(explicit) == 20 and set(manifest["exclusions"]["phase3_explicit_excluded_item_ids"]) == explicit
    phase4 = di.phase4_consumed_ids(PHASE4)
    exp019, provenance = di.exp019_item_ids(EXP019)
    assert provenance is not None and manifest["exclusions"]["exp019_items_excluded"] is True
    assert exp019["cifar100"] and exp019["eurosat"]
    for dataset_key, source in (("cifar100_test", "cifar100"), ("eurosat_sealed", "eurosat")):
        for role in ("development", "control_train"):
            ids = read_ids(out / f"exp021b__{dataset_key}__{role}.csv")
            assert all(role_by_id[(source, item_id)] == "reserved_not_accessed" for item_id in ids)
            assert not (set(ids) & explicit)
            assert not ({(source, item_id) for item_id in ids} & phase4)
            assert not (set(ids) & exp019[source])
    # the explicit exclusions are absent from every produced list, including 021A and the sealed list
    for path in out.glob("*.csv"):
        assert not (set(read_ids(path)) & explicit), path.name


def test_exp021a_and_021b_reserve_roles_do_not_overlap(real_only_build):
    out, _ = real_only_build
    calibration = set()
    for path in out.glob("exp021a__*.csv"):
        calibration |= set(read_ids(path))
    for path in out.glob("exp021b__*.csv"):
        assert not (set(read_ids(path)) & calibration), path.name


# ---------------------------------------------------------------- determinism and fail-closed


def test_byte_identical_regeneration(tmp_path, empty_root):
    require_real_inputs()
    out1, out2 = tmp_path / "one", tmp_path / "two"
    m1 = di.build(draw_args(out1, empty_root, skip_unavailable=True, allow_synthetic=False))
    m2 = di.build(draw_args(out2, empty_root, skip_unavailable=True, allow_synthetic=False))
    assert m1["_manifest_sha256"] == m2["_manifest_sha256"]
    names = sorted(path.name for path in out1.iterdir())
    assert names == sorted(path.name for path in out2.iterdir())
    for name in names:
        assert filecmp.cmp(out1 / name, out2 / name, shallow=False), name
    stored = json.loads((out1 / "item_manifest_ext.json").read_text(encoding="utf-8"))
    assert canonical_json_hash(stored) == m1["_manifest_sha256"] and stored["created_at"] == CREATED_AT


def test_fail_closed_without_skip_flag(tmp_path, empty_root):
    require_real_inputs()
    with pytest.raises(SystemExit) as info:
        di.build(draw_args(tmp_path / "out", empty_root, skip_unavailable=False, allow_synthetic=False))
    assert "FAIL CLOSED" in str(info.value) and "cifar100_test" in str(info.value)
    assert not (tmp_path / "out" / "item_manifest_ext.json").exists()


def test_official_check_rejects_synthetic_without_flag(tmp_path, synthetic_root):
    require_real_inputs()
    with pytest.raises(SystemExit) as info:
        di.build(draw_args(tmp_path / "out", synthetic_root, skip_unavailable=False, allow_synthetic=False))
    assert "MD5" in str(info.value)


def test_cli_rejects_missing_phase4_file(tmp_path):
    with pytest.raises(SystemExit):
        di.main(["--phase4-assignments", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "o")])
