"""Tests for the V2 fail-closed approval guard (V1 review finding P0-F).

  .venv/Scripts/python -m pytest satml2027ext/tests/test_guard_ext.py -q
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import candidates_ext, guard_ext  # noqa: E402
from satml2027ext._common import canonical_json_hash, sha256_file, write_json_atomic  # noqa: E402


def registration(experiment_id: str = "EXP-20260921-021B") -> dict:
    body = {"schema_version": guard_ext.REGISTRATION_SCHEMA, "experiment_id": experiment_id,
            "status": "preregistered_pending_independent_approval",
            "candidates": candidates_ext.canonical_candidates_block(), "cells": [{"cell_id": "x"}]}
    body["registration_sha256"] = canonical_json_hash(body)
    return body


class GuardTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(guard_ext, "verify_source_commit", return_value={"source_commit": "f" * 40})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.temp = Path(tempfile.mkdtemp())
        self.registrations = {experiment_id: registration(experiment_id) for experiment_id in guard_ext.EXPERIMENT_IDS}
        self.reg = self.registrations["EXP-20260921-021B"]
        self.preflight = self.temp / "preflight.json"
        self.manifest = self.temp / "manifest.json"
        self.custody = self.temp / "custody.json"
        for path in (self.preflight, self.manifest, self.custody):
            path.write_text(json.dumps({"file": path.name}), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def approval(self, **overrides) -> dict:
        body = guard_ext.approval_template({key: value["registration_sha256"] for key, value in self.registrations.items()})
        body.update({"status": guard_ext.REQUIRED_APPROVAL_STATUS, "preparation_authorized": True,
                     "approved_by": "reviewer", "approved_on": "2026-10-01", "source_commit": "f" * 40})
        body.pop("instructions")
        body.update(overrides)
        return body

    def write(self, body: dict) -> Path:
        path = self.temp / "approval.json"
        write_json_atomic(path, body)
        return path

    def check(self, body: dict, stage: str = "preparation", **files):
        return guard_ext.verify_approval(self.reg, stage=stage, approval_path=self.write(body), stage_files=files)

    def test_valid_preparation_approval_passes_and_template_authorizes_nothing(self):
        result = self.check(self.approval())
        self.assertEqual(result["status"], guard_ext.REQUIRED_APPROVAL_STATUS)
        template = guard_ext.approval_template({key: value["registration_sha256"] for key, value in self.registrations.items()})
        with self.assertRaises(guard_ext.ApprovalError):
            guard_ext.verify_approval(self.reg, stage="preparation", approval_path=self.write(template))

    def test_missing_flag_is_a_refusal_not_a_pass(self):
        for flag in ("preparation_authorized", "sampling_authorized", "analysis_authorized"):
            body = self.approval()
            del body[flag]
            with self.assertRaises(guard_ext.ApprovalError, msg=flag):
                self.check(body)
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(self.approval(preparation_authorized="true"))  # not a JSON boolean
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(self.approval(preparation_authorized=False))
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(self.approval(), stage="sampling")  # sampling flag false in this approval

    def test_status_schema_signature_and_unknown_keys(self):
        for overrides in ({"status": "DRAFT"}, {"approval_schema": "v1"}, {"approved_by": ""}, {"approved_on": " "},
                          {"surprise": 1}):
            with self.assertRaises(guard_ext.ApprovalError, msg=str(overrides)):
                self.check(self.approval(**overrides))

    def test_registrations_must_be_exactly_the_three_and_match(self):
        body = self.approval()
        body["registrations"].pop("EXP-20260921-021C")
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(body)
        body = self.approval()
        body["registrations"]["EXP-20260921-021B"] = "0" * 64
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(body)
        tampered = copy.deepcopy(self.reg)
        tampered["title"] = "changed after hashing"
        with self.assertRaises(RuntimeError):
            guard_ext.verify_approval(tampered, stage="preparation", approval_path=self.write(self.approval()))

    def test_dependency_map_must_equal_the_closure_exactly(self):
        closure = guard_ext.dependency_closure()
        self.assertIn("satml2027ext/run_shard_ext.py", closure)
        self.assertIn("satml2027/common/certify.py", closure)
        self.assertIn("interventions/boundary_active.py", closure)
        self.assertIn("certification/fss_baseline.py", closure)
        self.assertIn("configs/satml2027ext/exp-20260921-021b.json", closure)
        self.assertTrue(any(key.startswith("results/satml2027ext/items/") for key in closure))
        self.assertFalse(any("/tests/" in key for key in closure))
        body = self.approval()
        missing = dict(body["source_sha256"])
        missing.pop("satml2027ext/analyze_ext.py")
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(self.approval(source_sha256=missing))  # V1 silently skipped absent entries
        extra = dict(body["source_sha256"], **{"satml2027ext/ghost.py": "0" * 64})
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(self.approval(source_sha256=extra))
        changed = dict(body["source_sha256"], **{"satml2027/common/seeds.py": "0" * 64})
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(self.approval(source_sha256=changed))

    def test_sampling_and_analysis_bind_their_stage_files(self):
        body = self.approval(sampling_authorized=True)
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(body, stage="sampling", data_preflight=self.preflight, bank_manifest=self.manifest)
        body.update(data_preflight_sha256=sha256_file(self.preflight), bank_manifest_sha256=sha256_file(self.manifest))
        self.check(body, stage="sampling", data_preflight=self.preflight, bank_manifest=self.manifest)
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(body, stage="sampling", data_preflight=self.preflight)  # manifest file not supplied
        self.manifest.write_text("changed", encoding="utf-8")
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(body, stage="sampling", data_preflight=self.preflight, bank_manifest=self.manifest)
        body.update(bank_manifest_sha256=sha256_file(self.manifest), analysis_authorized=True)
        with self.assertRaises(guard_ext.ApprovalError):
            self.check(body, stage="analysis", data_preflight=self.preflight, bank_manifest=self.manifest,
                       run_custody_manifest=self.custody)
        body["run_custody_manifest_sha256"] = {"EXP-20260921-021B": sha256_file(self.custody)}
        self.check(body, stage="analysis", data_preflight=self.preflight, bank_manifest=self.manifest,
                   run_custody_manifest=self.custody)

    def test_v1_registrations_cannot_be_loaded(self):
        v1 = {**self.reg, "schema_version": "satml2027.registration.v2"}
        v1.pop("registration_sha256")
        v1["registration_sha256"] = canonical_json_hash(v1)
        path = self.temp / "v1.json"
        write_json_atomic(path, v1)
        with self.assertRaises(RuntimeError):
            guard_ext.load_registration(path)
        path = self.temp / "v2.json"
        write_json_atomic(path, self.reg)
        self.assertEqual(guard_ext.load_registration(path)["experiment_id"], "EXP-20260921-021B")


class V3GuardTests(unittest.TestCase):
    """Internal re-review of V2 (2026-09-26): n3 (duplicate keys, dates), M2b (preflight binding), m2 (021C parent custody)."""

    def setUp(self):
        patcher = mock.patch.object(guard_ext, "verify_source_commit", return_value={"source_commit": "f" * 40})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.temp = Path(tempfile.mkdtemp())
        self.registrations = {experiment_id: registration(experiment_id) for experiment_id in guard_ext.EXPERIMENT_IDS}
        self.files = {}
        for name in ("preflight", "other_preflight", "manifest", "custody", "parent_custody"):
            path = self.temp / f"{name}.json"
            path.write_text(json.dumps({"file": name}), encoding="utf-8")
            self.files[name] = path

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def approval(self, **overrides) -> dict:
        body = guard_ext.approval_template({key: value["registration_sha256"] for key, value in self.registrations.items()})
        body.update({"status": guard_ext.REQUIRED_APPROVAL_STATUS, "preparation_authorized": True,
                     "approved_by": "reviewer", "approved_on": "2026-10-01", "source_commit": "f" * 40})
        body.pop("instructions")
        body.update(overrides)
        return body

    def write_text(self, text: str) -> Path:
        path = self.temp / "approval.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_v3_n3_duplicate_keys_are_refused_in_both_orders(self):
        reg = self.registrations["EXP-20260921-021B"]
        body = json.dumps(self.approval(preparation_authorized=False))
        for extra in (', "preparation_authorized": true}', ', "preparation_authorized": false}'):
            with self.assertRaisesRegex(guard_ext.ApprovalError, "repeats a JSON key"):
                guard_ext.verify_approval(reg, stage="preparation", approval_path=self.write_text(body[:-1] + extra))

    def test_v3_n3_approved_on_must_be_an_iso_calendar_date(self):
        reg = self.registrations["EXP-20260921-021B"]
        for value in ("d", "01/10/2026", "2026-02-30"):
            with self.assertRaises(guard_ext.ApprovalError, msg=value):
                guard_ext.verify_approval(reg, stage="preparation",
                                          approval_path=self.write_text(json.dumps(self.approval(approved_on=value))))
        guard_ext.verify_approval(reg, stage="preparation", approval_path=self.write_text(json.dumps(self.approval())))

    def test_v3_m2b_a_named_preflight_binds_the_preparation_stage(self):
        reg = self.registrations["EXP-20260921-021B"]
        path = self.write_text(json.dumps(self.approval(data_preflight_sha256=sha256_file(self.files["preflight"]))))
        guard_ext.verify_approval(reg, stage="preparation", approval_path=path, stage_files={"data_preflight": self.files["preflight"]})
        with self.assertRaisesRegex(guard_ext.ApprovalError, "differs from the approved"):
            guard_ext.verify_approval(reg, stage="preparation", approval_path=path,
                                      stage_files={"data_preflight": self.files["other_preflight"]})
        with self.assertRaisesRegex(guard_ext.ApprovalError, "needs that file"):
            guard_ext.verify_approval(reg, stage="preparation", approval_path=path)

    def test_v3_m2_the_021c_analysis_needs_the_approved_parent_custody(self):
        parent = self.registrations["EXP-20260921-021B"]
        child = {"schema_version": guard_ext.REGISTRATION_SCHEMA, "experiment_id": "EXP-20260921-021C",
                 "status": "preregistered_pending_independent_approval", "cells": [{"cell_id": "x"}],
                 "candidates": candidates_ext.subset_candidates_block(candidates_ext.BUDGET_SUBSET_021C,
                                                                      parent_experiment_id=parent["experiment_id"],
                                                                      parent_registration_sha256=parent["registration_sha256"])}
        child["registration_sha256"] = canonical_json_hash(child)
        self.registrations["EXP-20260921-021C"] = child
        custody = {"EXP-20260921-021C": sha256_file(self.files["custody"]), "EXP-20260921-021B": sha256_file(self.files["parent_custody"])}
        body = self.approval(sampling_authorized=True, analysis_authorized=True,
                             data_preflight_sha256=sha256_file(self.files["preflight"]),
                             bank_manifest_sha256=sha256_file(self.files["manifest"]), run_custody_manifest_sha256=custody)
        path = self.write_text(json.dumps(body))
        files = {"data_preflight": self.files["preflight"], "bank_manifest": self.files["manifest"],
                 "run_custody_manifest": self.files["custody"]}
        with self.assertRaisesRegex(guard_ext.ApprovalError, "parent"):
            guard_ext.verify_approval(child, stage="analysis", approval_path=path, stage_files=files)
        guard_ext.verify_approval(child, stage="analysis", approval_path=path,
                                  stage_files={**files, "parent_run_custody_manifest": self.files["parent_custody"]})
        with self.assertRaisesRegex(guard_ext.ApprovalError, "parent run custody"):
            guard_ext.verify_approval(child, stage="analysis", approval_path=path,
                                      stage_files={**files, "parent_run_custody_manifest": self.files["other_preflight"]})


if __name__ == "__main__":
    unittest.main()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                             "-c", "core.autocrlf=false", *arguments], capture_output=True, check=True)
    return result.stdout.decode("ascii").strip()


@unittest.skipIf(shutil.which("git") is None, "git is not installed")
class SourceCommitTests(unittest.TestCase):
    """V4 (text review of reviewer bundle V3, item 22): the approved files must be committed with their exact bytes."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        _git(self.root, "init", "-q")
        (self.root / ".gitattributes").write_bytes(b"* -text\n")
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "a.py").write_bytes(b"print('a')\r\n")  # CRLF stays CRLF (* -text)
        (self.root / "pkg" / "b.json").write_bytes(b'{"b": 1}\n')
        _git(self.root, "add", ".gitattributes", "pkg/a.py", "pkg/b.json")
        _git(self.root, "commit", "-q", "-m", "files")
        self.commit = _git(self.root, "rev-parse", "HEAD")
        self.closure = {name: sha256_file(self.root / name) for name in ("pkg/a.py", "pkg/b.json")}

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_committed_files_pass_and_every_gap_is_refused(self):
        self.assertEqual(guard_ext.verify_source_commit(self.commit, self.closure, self.root)["files"], 2)
        self.assertEqual(guard_ext.git_blob_id(self.root / "pkg" / "a.py"), _git(self.root, "rev-parse", f"{self.commit}:pkg/a.py"))
        for bad in ("", "F" * 40, self.commit[:39], "0" * 40, 12):
            with self.assertRaises(guard_ext.ApprovalError):
                guard_ext.verify_source_commit(bad, self.closure, self.root)
        (self.root / "pkg" / "c.py").write_bytes(b"x = 1\n")  # approved but never committed
        with self.assertRaisesRegex(guard_ext.ApprovalError, "not committed"):
            guard_ext.verify_source_commit(self.commit, {**self.closure, "pkg/c.py": sha256_file(self.root / "pkg" / "c.py")}, self.root)
        (self.root / "pkg" / "b.json").write_bytes(b'{"b": 2}\n')  # changed after the commit
        with self.assertRaisesRegex(guard_ext.ApprovalError, "different bytes"):
            guard_ext.verify_source_commit(self.commit, self.closure, self.root)
        _git(self.root, "add", "pkg/b.json", "pkg/c.py")
        _git(self.root, "commit", "-q", "-m", "update")
        later = _git(self.root, "rev-parse", "HEAD")
        guard_ext.verify_source_commit(later, {name: sha256_file(self.root / name) for name in ("pkg/a.py", "pkg/b.json", "pkg/c.py")},
                                       self.root)
        with self.assertRaises(guard_ext.ApprovalError):  # the older commit no longer holds these bytes
            guard_ext.verify_source_commit(self.commit, self.closure, self.root)

    def test_verify_approval_calls_the_commit_check(self):
        registrations = {experiment_id: registration(experiment_id) for experiment_id in guard_ext.EXPERIMENT_IDS}
        body = guard_ext.approval_template({key: value["registration_sha256"] for key, value in registrations.items()})
        body.update({"status": guard_ext.REQUIRED_APPROVAL_STATUS, "preparation_authorized": True,
                     "approved_by": "reviewer", "approved_on": "2026-10-01", "source_commit": "0" * 40})
        body.pop("instructions")
        path = self.root / "approval.json"
        write_json_atomic(path, body)
        with self.assertRaisesRegex(guard_ext.ApprovalError, "source_commit"):
            guard_ext.verify_approval(registrations["EXP-20260921-021B"], stage="preparation", approval_path=path)
