#!/usr/bin/env python3
"""Recompute the custody chain of the EXP-019A/B and N3C result download.

Writes ``custody.json`` next to this script.  Every hash is recomputed from the
local download; nothing is copied from a receipt without being compared with a
recomputed value, and every comparison carries an explicit match flag.

Usage (from the repository root, or from an extracted review bundle whose
``results/satml2027_downloads/...`` tree is present):

  python artifacts/satml2027/EXP019_RESULT_REVIEW_20260925_V1/build_custody.py
      [--download-root PATH] [--copy2 PATH] [--out custody.json]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from review_common import (
    DOWNLOAD_COPY2,
    DOWNLOAD_ROOT,
    PACKAGE_DIR,
    PACKAGE_NAME,
    REGISTRATION_FILES,
    load_registration,
    parse_manifest,
    parse_receipt,
    read_json,
    relativize,
    server_to_local,
    sha256_file,
    utc_now,
    write_json,
)

# Values quoted in the decision log / research notes (not in the download).
# They are compared with recomputed values and never trusted on their own.
D120_ATTEMPT3_SHARDS = {
    "openai-clip-vit-l14-quickgelu__cifar100__sigma0.12": (
        "a4802a7d8ea5bd099f13c17d9c86e37153eaa999802f972c49f1cc6c12b3b32b",
        "6b2b4f962e82aed522b83fc1c8396c280016ee1c81c3cddfa864b09218354a0a",
    ),
    "openai-clip-vit-l14-quickgelu__cifar100__sigma0.5": (
        "dad3c7ecc971d6a9d04fda16ed185686db0d435baaec68811684748eb4aa1194",
        "c4d76877f8c99a6b22d09beff2a7df9c48c4273bb8c6689907a9c658ffc79b2e",
    ),
    "openai-clip-vit-l14-quickgelu__cifar10__sigma0.12": (
        "eed370c7b7378346fe631b6ce97f7cf4a54ec65b9207ce25c580aca638600540",
        "7e1125d063e4e0dac2ffb91aceb72148ec505bf9f7bd8ffbf37d5168f53cb12f",
    ),
    "openai-clip-vit-l14-quickgelu__cifar10__sigma0.25": (
        "0a89a17ce66c495c01710cf41185ef3e4df1202f3c8a1817eb69a93c5d485e05",
        "e0fe55e0d31f220942e6723720fc68fd60dae097fc4aaffee43509beb74f97c4",
    ),
    "openai-clip-vit-l14-quickgelu__cifar10__sigma0.5": (
        "52549541e25e993c22f3f23c4a78ee2a4dcc7621200065121c606f27a9922661",
        "b153e5524d812e05dfbfae9b9d19bfb9b45a0dea9d0ef6e1b7724b422111a478",
    ),
}
D118_ATTEMPT3_RECEIPT_SHA256 = "95f6323c3bdb929fce56c34465b7f1a58dd5e43197d87f99808fc7418b559c8a"
D118_ATTEMPT3_LOG_SHA256 = "a9494544b6a74de71a71aa585eef420e93fc54e56dfa7646861b30462155c41d"
D119_RECOVERY_GUARD_V2_SHA256 = "ef8a3a6d450c03587122602a3f18baed33f3889136e99e8bc768b46c70e908f6"
D123_CHECKPOINT_HASHES = {
    "openai-clip-vit-b32-quickgelu": ("timm/vit_base_patch32_clip_224.openai @ a6f597a3",
                                      "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"),
    "openai-clip-vit-b16-quickgelu": ("timm/vit_base_patch16_clip_224.openai @ 977e3dd0",
                                      "4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f"),
    "openai-clip-rn50-quickgelu": ("timm/resnet50_clip.openai @ ec3d92cf",
                                   "da0baa37fb2211eee5729ab69ed587eb10d48f5b7076ceeed01552fc3d4cb4ee"),
    "openclip-vit-b32-laion2b": ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K @ 1a25a446",
                                 "ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6"),
    "openai-clip-vit-l14-quickgelu": ("timm/vit_large_patch14_clip_224.openai @ 18d05354",
                                      "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"),
}
APPROVED_COMMIT = "7c26704eb3899893dd385ae3feeace97b078c41a"


def hash_entry(root: Path, relative: str) -> dict:
    path = root / relative
    if not path.is_file():
        return {"present": False, "sha256": None, "bytes": None}
    return {"present": True, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify_manifest(manifest_path: Path, root: Path, transform=None) -> dict:
    entries = parse_manifest(manifest_path)
    rows, mismatches, missing = [], 0, 0
    for expected, listed in entries:
        relative = transform(listed) if transform else listed
        recomputed = hash_entry(root, relative)
        match = recomputed["present"] and recomputed["sha256"] == expected
        if not recomputed["present"]:
            missing += 1
        elif not match:
            mismatches += 1
        rows.append({"listed_path": listed, "local_path": relative, "expected_sha256": expected,
                     "recomputed_sha256": recomputed["sha256"], "bytes": recomputed["bytes"], "match": bool(match)})
    return {
        "manifest_file": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "entry_count": len(entries),
        "matched": len(entries) - mismatches - missing,
        "mismatched": mismatches,
        "missing": missing,
        "all_match": mismatches == 0 and missing == 0,
        "files": rows,
    }


def launcher_jobs(launcher_path: Path) -> list[dict]:
    """Job list of the approved launcher: (index, cell, registration, out dir, command block)."""

    text = launcher_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    jobs = []
    current = None
    for line_number, line in enumerate(lines, start=1):
        header = re.match(r'^echo "\[\$\(date -Is\)\] (\S+) shard (\d+)/(\d+)"$', line)
        if header:
            current = {"job": len(jobs) + 1, "cell_id": header.group(1), "shard": f"{header.group(2)}/{header.group(3)}",
                       "first_line": line_number, "lines": [line]}
            jobs.append(current)
            continue
        if current is not None:
            current["lines"].append(line)
            if line.strip().startswith("2>&1 | tee -a"):
                current["last_line"] = line_number
                current["command"] = "\n".join(current["lines"])
                match = re.search(r"--registration (\S+)", current["command"])
                current["registration"] = match.group(1) if match else None
                match = re.search(r"--out (\S+)", current["command"])
                current["out"] = match.group(1) if match else None
                current["allow_tf32_flag_present"] = "--allow-tf32" in current["command"]
                current["store_features"] = "--store-features" in current["command"]
                current = None
    for job in jobs:
        job.pop("lines", None)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, default=DOWNLOAD_ROOT)
    parser.add_argument("--copy2", type=Path, default=DOWNLOAD_COPY2)
    parser.add_argument("--out", type=Path, default=PACKAGE_DIR / "custody.json")
    parser.add_argument("--skip-full-tree", action="store_true",
                        help="verify only the 76 registered result files, not the whole 411-file download manifest")
    args = parser.parse_args()
    root: Path = args.download_root
    server = root / "server_results_root"
    results = root / "scientific_results"

    receipts = {}
    for name in ("scientific_run_attempt4_receipt.txt", "scientific_run_attempt3_receipt.txt",
                 "scientific_run_receipt.txt", "day1_prepare_attempt2_receipt.txt",
                 "scientific_run_segfault_recovery_v2_record.txt", "scientific_run_reboot_recovery_consumed_v1.txt"):
        path = server / name
        receipts[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
                          "fields": parse_receipt(path)}
    attempt4 = receipts["scientific_run_attempt4_receipt.txt"]["fields"]
    attempt3 = receipts["scientific_run_attempt3_receipt.txt"]["fields"]
    attempt1 = receipts["scientific_run_receipt.txt"]["fields"]
    prepare = receipts["day1_prepare_attempt2_receipt.txt"]["fields"]
    record = receipts["scientific_run_segfault_recovery_v2_record.txt"]["fields"]
    reboot = receipts["scientific_run_reboot_recovery_consumed_v1.txt"]["fields"]

    # 1. The 76 registered result files of the attempt-4 manifest.
    result_manifest = server / "scientific_run_attempt4_receipt.txt.manifest.sha256"
    registered = verify_manifest(result_manifest, root, transform=server_to_local)
    registered["receipt_result_file_count"] = int(attempt4["result_file_count"])
    registered["entry_count_matches_receipt"] = registered["entry_count"] == int(attempt4["result_file_count"])
    registered["receipt_result_manifest_sha256"] = attempt4["result_manifest_sha256"]
    registered["manifest_file_sha256_matches_receipt"] = registered["manifest_file_sha256"] == attempt4["result_manifest_sha256"]
    copy2 = None
    if args.copy2.is_dir():
        copy2 = verify_manifest(result_manifest, args.copy2, transform=server_to_local)
        copy2.pop("files")
        copy2["root"] = str(args.copy2)
    by_experiment: dict[str, int] = {}
    for row in registered["files"]:
        experiment = row["listed_path"].split("/")[2]
        by_experiment[experiment] = by_experiment.get(experiment, 0) + 1
    registered["files_per_experiment"] = by_experiment

    # 2. Attempt-3 shard custody snapshot (D-120) against the recovery record and the recomputed values.
    attempt3_shards = []
    for cell_id, (doc_shard, doc_meta) in D120_ATTEMPT3_SHARDS.items():
        base = f"scientific_results/EXP-20260920-019A/cells/{cell_id}/shard_000_of_001"
        shard = hash_entry(root, base + ".npz")
        meta = hash_entry(root, base + ".meta.json")
        record_shard = record.get(f"completed_shard_sha256[{cell_id}]")
        record_meta = record.get(f"completed_meta_sha256[{cell_id}]")
        manifest_shard = next((r["expected_sha256"] for r in registered["files"] if r["local_path"] == base + ".npz"), None)
        manifest_meta = next((r["expected_sha256"] for r in registered["files"] if r["local_path"] == base + ".meta.json"), None)
        attempt3_shards.append({
            "cell_id": cell_id,
            "shard_sha256_recomputed": shard["sha256"], "shard_bytes": shard["bytes"],
            "shard_sha256_recovery_record": record_shard, "shard_sha256_d120_note": doc_shard,
            "shard_sha256_attempt4_manifest": manifest_shard,
            "shard_match_record": shard["sha256"] == record_shard,
            "shard_match_d120_note": shard["sha256"] == doc_shard,
            "shard_match_attempt4_manifest": shard["sha256"] == manifest_shard,
            "meta_sha256_recomputed": meta["sha256"], "meta_bytes": meta["bytes"],
            "meta_sha256_recovery_record": record_meta, "meta_sha256_d120_note": doc_meta,
            "meta_sha256_attempt4_manifest": manifest_meta,
            "meta_match_record": meta["sha256"] == record_meta,
            "meta_match_d120_note": meta["sha256"] == doc_meta,
            "meta_match_attempt4_manifest": meta["sha256"] == manifest_meta,
        })
    attempt3_all_match = all(all(row[key] for key in row if key.endswith(("_match_record", "_match_d120_note", "_match_attempt4_manifest")))
                             for row in attempt3_shards)

    # 3. Plan / launcher / registrations / preparation manifest / recovery record hashes.
    plan = hash_entry(root, "scientific_results/plan/plan.json")
    launcher = hash_entry(root, "scientific_results/plan/launch/run_serverA_gpu0.sh")
    prep_manifest_path = server / "day1_prepare_attempt2_receipt.txt.manifest.sha256"
    prep_manifest_hash = sha256_file(prep_manifest_path)
    prep_verify = verify_manifest(prep_manifest_path, root, transform=server_to_local)
    prep_verify.pop("files")
    frozen = {
        "plan_json": {
            "recomputed_sha256": plan["sha256"], "bytes": plan["bytes"],
            "attempt4_receipt": attempt4["plan_sha256"], "attempt3_receipt": attempt3["plan_sha256"],
            "attempt1_receipt": attempt1["plan_sha256"], "recovery_record": record["plan_sha256"],
            "reboot_consumption_record": reboot["plan_sha256"],
            "match_all": all(plan["sha256"] == v for v in (attempt4["plan_sha256"], attempt3["plan_sha256"],
                                                          attempt1["plan_sha256"], record["plan_sha256"], reboot["plan_sha256"])),
        },
        "approved_launcher": {
            "recomputed_sha256": launcher["sha256"], "bytes": launcher["bytes"],
            "attempt4_receipt": attempt4["approved_launcher_sha256"], "attempt3_receipt": attempt3["launcher_sha256"],
            "attempt1_receipt": attempt1["launcher_sha256"], "recovery_record": record["approved_launcher_sha256"],
            "reboot_consumption_record": reboot["launcher_sha256"],
            "match_all": all(launcher["sha256"] == v for v in (attempt4["approved_launcher_sha256"], attempt3["launcher_sha256"],
                                                              attempt1["launcher_sha256"], record["approved_launcher_sha256"],
                                                              reboot["launcher_sha256"])),
        },
        "preparation_manifest": {
            "recomputed_sha256": prep_manifest_hash, "entry_count": prep_verify["entry_count"],
            "attempt4_receipt": attempt4["preparation_manifest_sha256"], "prepare_receipt": prepare["manifest_sha256"],
            "recovery_record": record["preparation_manifest_sha256"], "reboot_consumption_record": reboot["preparation_manifest_sha256"],
            "match_all": all(prep_manifest_hash == v for v in (attempt4["preparation_manifest_sha256"], prepare["manifest_sha256"],
                                                              record["preparation_manifest_sha256"], reboot["preparation_manifest_sha256"])),
            "prepare_receipt_counts": {k: v for k, v in prepare.items() if k.endswith("_count")},
            "local_verification_of_252_listed_files": prep_verify,
        },
        "recovery_record": {
            "recomputed_sha256": receipts["scientific_run_segfault_recovery_v2_record.txt"]["sha256"],
            "attempt4_receipt": attempt4["recovery_record_sha256"],
            "match": receipts["scientific_run_segfault_recovery_v2_record.txt"]["sha256"] == attempt4["recovery_record_sha256"],
        },
        "reboot_consumption_record": {
            "recomputed_sha256": receipts["scientific_run_reboot_recovery_consumed_v1.txt"]["sha256"],
            "attempt3_receipt": attempt3["recovery_consumption_sha256"],
            "match": receipts["scientific_run_reboot_recovery_consumed_v1.txt"]["sha256"] == attempt3["recovery_consumption_sha256"],
        },
        "attempt3_receipt": {
            "recomputed_sha256": receipts["scientific_run_attempt3_receipt.txt"]["sha256"],
            "d118_note": D118_ATTEMPT3_RECEIPT_SHA256,
            "match": receipts["scientific_run_attempt3_receipt.txt"]["sha256"] == D118_ATTEMPT3_RECEIPT_SHA256,
        },
        "attempt3_log": {
            "recomputed_sha256": sha256_file(server / "scientific_run_attempt3.log"),
            "d118_note": D118_ATTEMPT3_LOG_SHA256,
            "match": sha256_file(server / "scientific_run_attempt3.log") == D118_ATTEMPT3_LOG_SHA256,
        },
    }
    attempt1_hashes = verify_manifest(server / "scientific_run_attempt1_failure_hashes.sha256", server,
                                      transform=lambda p: p.rsplit("/", 1)[-1])
    frozen["attempt1_failure_hashes"] = {k: v for k, v in attempt1_hashes.items() if k != "files"} | {
        "files": [{k: r[k] for k in ("listed_path", "recomputed_sha256", "match")} for r in attempt1_hashes["files"]]}

    registrations = {}
    for experiment_id, filename in REGISTRATION_FILES.items():
        payload, provenance = load_registration(experiment_id)
        sidecar = provenance["path"].replace(".json", ".sha256")
        sidecar_value = Path(sidecar).read_text(encoding="utf-8").strip() if Path(sidecar).is_file() else None
        cells_dir = results / experiment_id / "cells"
        sidecar_hashes = set()
        for meta_path in sorted(cells_dir.glob("*/shard_*_of_*.meta.json")):
            sidecar_hashes.add(read_json(meta_path).get("registration_sha256"))
        registrations[experiment_id] = {
            **provenance,
            "sidecar_file": sidecar, "sidecar_value": sidecar_value,
            "sidecar_matches_recomputed": sidecar_value == provenance["registration_sha256_recomputed"],
            "shard_sidecars_registration_sha256": sorted(h for h in sidecar_hashes if h),
            "shard_sidecars_all_match": sidecar_hashes == {provenance["registration_sha256_recomputed"]},
            "registered_cell_count": len(payload["cells"]),
        }

    # 4. Resume launcher derivation.
    jobs = launcher_jobs(root / "scientific_results/plan/launch/run_serverA_gpu0.sh")
    guard_console = (server / "scientific_run_attempt4_guard_console.log").read_text(encoding="utf-8")
    guard_match = re.search(r"resume launcher written: (\d+) remaining jobs; sha256 ([0-9a-f]{64})", guard_console)
    attempt3_log_lines = (server / "scientific_run_attempt3.log").read_text(encoding="utf-8").splitlines()
    attempt3_started = [re.match(r"^\[[^\]]+\] (\S+) shard", line).group(1) for line in attempt3_log_lines
                        if re.match(r"^\[[^\]]+\] (\S+) shard", line)]
    attempt3_done = [line.split("cells/")[1].split("/")[0] for line in attempt3_log_lines if line.startswith("done: ")]
    attempt3_segfault = [line for line in attempt3_log_lines if "Segmentation fault" in line]
    attempt4_log_lines = (server / "scientific_run_attempt4.log").read_text(encoding="utf-8").splitlines()
    attempt4_started = [re.match(r"^\[[^\]]+\] (\S+) shard", line).group(1) for line in attempt4_log_lines
                        if re.match(r"^\[[^\]]+\] (\S+) shard", line)]
    attempt4_done = [line.split("cells/")[1].split("/")[0] for line in attempt4_log_lines if line.startswith("done: ")]
    resume_hash_sources = {
        "attempt4_receipt": attempt4.get("resume_launcher_sha256"),
        "recovery_record": record.get("resume_launcher_sha256"),
        "guard_console": guard_match.group(2) if guard_match else None,
    }
    job6 = jobs[5]
    resume = {
        "approved_launcher_sha256_recomputed": launcher["sha256"],
        "approved_launcher_sha256_receipts_match": frozen["approved_launcher"]["match_all"],
        "approved_launcher_job_count": len(jobs),
        "approved_launcher_jobs": [{k: v for k, v in job.items() if k != "command"} for job in jobs],
        "worker_commands_pass_allow_tf32": any(job["allow_tf32_flag_present"] for job in jobs),
        "attempt3_completed_jobs": attempt3_done,
        "attempt3_completed_jobs_equal_launcher_jobs_1_to_5": attempt3_done == [job["cell_id"] for job in jobs[:5]],
        "attempt3_job_started_but_not_completed": [c for c in attempt3_started if c not in attempt3_done],
        "attempt3_segfault_lines": attempt3_segfault,
        "job6_cell_id": job6["cell_id"],
        "job6_verbatim_command": job6["command"],
        "job6_launcher_lines": [job6["first_line"], job6["last_line"]],
        "resume_launcher_path_on_server": record.get("resume_launcher"),
        "resume_launcher_sha256_sources": resume_hash_sources,
        "resume_launcher_sha256_sources_agree": len({v for v in resume_hash_sources.values() if v}) == 1,
        "resume_launcher_job_count_guard_console": int(guard_match.group(1)) if guard_match else None,
        "resume_launcher_job_count_expected": len(jobs) - 5,
        "resume_launcher_job_count_consistent": bool(guard_match) and int(guard_match.group(1)) == len(jobs) - 5,
        "resume_launcher_local_copy_available": False,
        "resume_launcher_note": ("the resume launcher itself is not part of the downloaded result chain; its hash is "
                                 "quoted by three server records (receipt, recovery record, guard console) and the guard "
                                 "asserted that its job commands are the verbatim suffix of the approved launcher from job 6; "
                                 "the header substitution (absolute source root) prevents re-deriving the hash locally"),
        "attempt4_started_jobs": attempt4_started,
        "attempt4_completed_jobs": attempt4_done,
        "attempt4_first_job_is_job6": bool(attempt4_started) and attempt4_started[0] == job6["cell_id"],
        "attempt4_jobs_equal_launcher_jobs_6_to_36": attempt4_done == [job["cell_id"] for job in jobs[5:]],
        "attempt3_plus_attempt4_cover_all_jobs_once": attempt3_done + attempt4_done == [job["cell_id"] for job in jobs],
        "guard_console_text": guard_console,
        "recovery_guard_v2_sha256_from_d119_note": D119_RECOVERY_GUARD_V2_SHA256,
        "recovery_guard_v2_local_copy_available": False,
    }

    # 5. Whole-download manifest (411 files) and the remote scientific_results manifest.
    download_manifest = None
    remote_manifest = None
    if not args.skip_full_tree:
        download_manifest = verify_manifest(root / "MANIFEST.sha256", root)
        download_manifest["files"] = [r for r in download_manifest["files"] if not r["match"]]
        download_manifest["note"] = "only non-matching entries are listed under files"
        remote_manifest = verify_manifest(root / "REMOTE_scientific_results.sha256", results)
        remote_manifest["files"] = [r for r in remote_manifest["files"] if not r["match"]]
        remote_manifest["note"] = "remote-side sha256sum of the server results tree; only non-matching entries are listed"

    transfer = read_json(root / "incoming" / "TRANSFER_MANIFEST.json")
    incoming = []
    for entry in transfer["files"]:
        local = hash_entry(root / "incoming", entry["path"])
        incoming.append({"path": entry["path"], "expected_sha256": entry["sha256"], "recomputed_sha256": local["sha256"],
                         "bytes": local["bytes"], "match": local["sha256"] == entry["sha256"]})

    custody = {
        "schema_version": "exp019_result_review.custody.v1",
        "package": PACKAGE_NAME,
        "generated_utc": utc_now(),
        "download_root": str(root),
        "approved_commit": APPROVED_COMMIT,
        "approved_commit_matches_transfer_manifest": transfer.get("approved_commit") == APPROVED_COMMIT,
        "summary": {
            "registered_result_files": registered["entry_count"],
            "registered_result_files_match": registered["matched"],
            "registered_result_files_all_match": registered["all_match"],
            "result_manifest_sha256_matches_receipt": registered["manifest_file_sha256_matches_receipt"],
            "copy2_all_match": copy2["all_match"] if copy2 else None,
            "attempt3_five_shards_all_match": attempt3_all_match,
            "plan_match_all": frozen["plan_json"]["match_all"],
            "approved_launcher_match_all": frozen["approved_launcher"]["match_all"],
            "preparation_manifest_match_all": frozen["preparation_manifest"]["match_all"],
            "preparation_files_all_match": prep_verify["all_match"],
            "recovery_record_match": frozen["recovery_record"]["match"],
            "registrations_all_match": all(r["registration_sha256_match"] and r["sidecar_matches_recomputed"]
                                           and r["shard_sidecars_all_match"] for r in registrations.values()),
            "resume_launcher_sources_agree": resume["resume_launcher_sha256_sources_agree"],
            "resume_launcher_job_count_consistent": resume["resume_launcher_job_count_consistent"],
            "attempt3_plus_attempt4_cover_all_jobs_once": resume["attempt3_plus_attempt4_cover_all_jobs_once"],
            "download_manifest_all_match": download_manifest["all_match"] if download_manifest else None,
            "remote_manifest_all_match": remote_manifest["all_match"] if remote_manifest else None,
            "incoming_transfer_all_match": all(e["match"] for e in incoming),
        },
        "receipts": receipts,
        "registered_result_files": registered,
        "registered_result_files_copy2": copy2,
        "attempt3_shard_custody_snapshot": {"all_match": attempt3_all_match, "cells": attempt3_shards},
        "frozen_inputs": frozen,
        "registrations": registrations,
        "resume_launcher_derivation": resume,
        "download_manifest": download_manifest,
        "remote_scientific_results_manifest": remote_manifest,
        "incoming_transfer": {"approved_commit": transfer.get("approved_commit"), "files": incoming},
        "checkpoint_binding": {
            "note": ("EXP-019 binds checkpoints by open_clip name and pretrained tag, not by file hash (D-122). The five "
                     "checkpoint blob hashes below were recorded when the checkpoints were staged into the worker cache on "
                     "2026-09-25 (D-123, research/SATML2027_SEGFAULT_RECOVERY_20260925.md); they are not verifiable from "
                     "this download and are listed for disclosure only."),
            "hashes": {model: {"snapshot": snap, "sha256": digest} for model, (snap, digest) in D123_CHECKPOINT_HASHES.items()},
        },
    }
    all_ok = all(v is True for k, v in custody["summary"].items() if isinstance(v, bool))
    custody["summary"]["all_checks_pass"] = all_ok
    custody = relativize(custody)
    write_json(args.out, custody)
    print(f"custody: {registered['matched']}/{registered['entry_count']} registered files match; "
          f"all checks pass = {all_ok}; wrote {args.out}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
