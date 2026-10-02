#!/usr/bin/env python3
"""Build manifest.sha256.json for the frozen EXP-20260917-020-FARLA-FULL result tree.

The manifest records the SHA-256 digest and byte size of every file under the
downloaded result directory (recursively), the configuration digest recorded in
CONFIG_SHA256, and whether config_snapshot.json hashes to that digest.

FARLA stores the configuration digest as the SHA-256 of the *canonical JSON*
form of the configuration (keys sorted, separators ",", ":", ASCII-only), not
as the digest of the file bytes. Both digests are reported so a reviewer can
see the difference. This script imports nothing from the FARLA package.

Run from the repository root:
    .venv/Scripts/python FARLA/artifacts/phase4/FARLA_FULL_RESULT_CUSTODY_V1/build_manifest.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent
FARLA_ROOT = PACKAGE_DIR.parents[2]
REPO_ROOT = PACKAGE_DIR.parents[3]
DEFAULT_RESULTS = FARLA_ROOT / "downloaded_results" / "EXP-20260917-020-FARLA-FULL"
DEFAULT_REPO_CONFIG = FARLA_ROOT / "configs" / "phase4" / "farla_full_v1.json"
SCHEMA_VERSION = "farla_result_custody.manifest.v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def describe_config_file(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    raw = path.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    return {
        "exists": True,
        "bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_json_sha256": canonical_json_sha256(parsed),
        "experiment_id": parsed.get("experiment_id"),
    }


def build_manifest(results_dir: Path, repo_config: Path) -> dict:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory is absent: {results_dir}")
    files = []
    total_bytes = 0
    for path in sorted(p for p in results_dir.rglob("*") if p.is_file()):
        stat = path.stat()
        total_bytes += stat.st_size
        files.append(
            {
                "path": path.relative_to(results_dir).as_posix(),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                # Informational only: modification times change when files are copied.
                "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
            }
        )
    recorded = (results_dir / "CONFIG_SHA256").read_text(encoding="utf-8").strip()
    snapshot = describe_config_file(results_dir / "config_snapshot.json")
    repo = describe_config_file(repo_config)
    snapshot_matches = snapshot.get("canonical_json_sha256") == recorded
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "EXP-20260917-020-FARLA-FULL",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "script": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results_dir": results_dir.relative_to(REPO_ROOT).as_posix(),
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "config": {
            "recorded_config_sha256": recorded,
            "recorded_from": "CONFIG_SHA256",
            "digest_definition": (
                "SHA-256 of json.dumps(config, sort_keys=True, separators=(',', ':'), "
                "ensure_ascii=True) as used by the FARLA configuration loader"
            ),
            "config_snapshot": {
                "path": "config_snapshot.json",
                **snapshot,
                "canonical_sha256_matches_recorded": snapshot_matches,
                "file_sha256_matches_recorded": snapshot.get("file_sha256") == recorded,
            },
            "repository_config": {
                "path": repo_config.relative_to(REPO_ROOT).as_posix(),
                **repo,
                "byte_identical_to_snapshot": (
                    repo.get("file_sha256") is not None
                    and repo.get("file_sha256") == snapshot.get("file_sha256")
                ),
                "canonical_sha256_matches_recorded": repo.get("canonical_json_sha256") == recorded,
            },
        },
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--repo-config", default=str(DEFAULT_REPO_CONFIG))
    parser.add_argument("--out", default=str(PACKAGE_DIR / "manifest.sha256.json"))
    args = parser.parse_args()
    manifest = build_manifest(Path(args.results).resolve(), Path(args.repo_config).resolve())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    config = manifest["config"]
    print(f"wrote {out}")
    print(f"files={manifest['file_count']} total_bytes={manifest['total_bytes']}")
    print(f"recorded CONFIG_SHA256={config['recorded_config_sha256']}")
    print(
        "config_snapshot canonical hash matches recorded: "
        f"{config['config_snapshot']['canonical_sha256_matches_recorded']}"
    )
    print(
        "repository config byte-identical to snapshot: "
        f"{config['repository_config']['byte_identical_to_snapshot']}"
    )


if __name__ == "__main__":
    main()
