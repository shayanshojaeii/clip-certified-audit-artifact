#!/usr/bin/env python3
"""Zip the custody package and write the adjacent external manifest.

Members are stored under repository-relative paths, so extracting the archive
at the repository root recreates FARLA/artifacts/phase4/FARLA_FULL_RESULT_CUSTODY_V1/
and tests/test_farla_result_custody_v1.py in place. The external manifest
(FARLA_FULL_RESULT_CUSTODY_V1.external_manifest.json, next to the zip) records
the zip's SHA-256, size and every member's SHA-256; it is authoritative for the
delivered archive bytes.

Run from the repository root:
    .venv/Scripts/python FARLA/artifacts/phase4/FARLA_FULL_RESULT_CUSTODY_V1/build_package_zip.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[3]
ZIP_PATH = PACKAGE_DIR.parent / f"{PACKAGE_DIR.name}.zip"
EXTERNAL_PATH = PACKAGE_DIR.parent / f"{PACKAGE_DIR.name}.external_manifest.json"
EXTRA_FILES = [REPO_ROOT / "tests" / "test_farla_result_custody_v1.py"]
REQUIRED_PACKAGE_FILES = [
    "START_HERE.md",
    "manifest.sha256.json",
    "build_manifest.py",
    "reconstruct_farla_results.py",
    "reconstruction_report.json",
    "reconstruction_report.md",
    "build_package_zip.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files() -> list[Path]:
    missing = [name for name in REQUIRED_PACKAGE_FILES if not (PACKAGE_DIR / name).is_file()]
    missing += [str(path) for path in EXTRA_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot build the package zip; missing: {missing}")
    files = [
        path for path in sorted(PACKAGE_DIR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    return files + list(EXTRA_FILES)


def main() -> None:
    files = collect_files()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    members = []
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            arcname = path.relative_to(REPO_ROOT).as_posix()
            archive.write(path, arcname)
            members.append({"arcname": arcname, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = sorted(archive.namelist())
        if names != sorted(member["arcname"] for member in members):
            raise RuntimeError("zip member list does not match the collected files")
        for member in members:
            if hashlib.sha256(archive.read(member["arcname"])).hexdigest() != member["sha256"]:
                raise RuntimeError(f"zip member does not read back identically: {member['arcname']}")
    external = {
        "schema_version": "farla_result_custody.external_manifest.v1",
        "experiment_id": "EXP-20260917-020-FARLA-FULL",
        "package": PACKAGE_DIR.name,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": {"script": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                    "python": sys.version.split()[0]},
        "extract_at": "repository root (members carry repository-relative paths)",
        "zip": {
            "path": ZIP_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(ZIP_PATH),
            "bytes": ZIP_PATH.stat().st_size,
            "member_count": len(members),
        },
        "members": members,
    }
    EXTERNAL_PATH.write_text(json.dumps(external, indent=2) + "\n", encoding="utf-8")
    print(f"zip     : {ZIP_PATH}")
    print(f"sha256  : {external['zip']['sha256']}")
    print(f"bytes   : {external['zip']['bytes']}")
    print(f"members : {len(members)}")
    for member in members:
        print(f"  {member['arcname']}  {member['bytes']} B  {member['sha256'][:16]}...")
    print(f"external manifest: {EXTERNAL_PATH}")


if __name__ == "__main__":
    main()
