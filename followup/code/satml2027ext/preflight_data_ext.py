#!/usr/bin/env python3
"""EXP-021 data and environment preflight (V2; V1 review finding "data-root rule not enforced").

Run once on the sampling host, before the preparation stage.  It writes
``EXP021_DATA_PREFLIGHT.json`` with status PASS only if every check passes:

1. the three source archives hash to the values in the registrations'
   ``data_roots`` block (CIFAR-100 tarball MD5 and SHA-256, Imagenette-320 and
   EuroSAT archives SHA-256);
2. the unpacked data root is the one the loaders verify: the CIFAR-100 train,
   test and meta batches have torchvision's MD5 values, the Imagenette split
   counts and class folders are official, every EuroSAT image a registered list
   names exists;
3. ``draw_items_ext.py`` run from this data root with the registered
   ``created_at`` reproduces every registered item list and the item manifest
   byte for byte;
4. every image file that a registered list names (EuroSAT and Imagenette, all
   roles) is hashed and compared with the SHA-256 of its member in the
   registered archive (streamed, never decoded; V3, re-review M2a), so the
   unpacked data root is tied to the archive bytes; the worker and the
   preparation stage later require each file they load to have exactly this
   hash;
5. the runtime environment (Python and package versions, the SHA-256 of each
   key package's installed RECORD file, CUDA, device) is recorded; the worker
   and the preparation stage refuse to run in a different environment.

The preflight is a custody step, not a scientific stage: it reads labels and
file bytes, never decodes an image, computes no model output and writes no
scientific output.  Hashing the bytes of a sealed EuroSAT file is therefore not
sealed evaluation access, which the registrations define as decoding a sealed
image or computing any model output on it.  The preflight may run before the
stage-1 approval; a stage-1 approval that names its hash binds the preparation
stage to it, the bank manifest records the preflight the preparation used, and
the sampling approval binds the same file (the worker refuses a bank manifest
built from any other preflight).  Regenerating the item lists needs the Phase-3
and Phase-4 assignment files and the EXP-019 item lists on the host
(``draw_items_ext.DEFAULT_PHASE3``/``DEFAULT_PHASE4``/``DEFAULT_EXP019_ITEMS``);
a missing input is a failed check, not a crash.

Usage:
  HF_HUB_OFFLINE=1 python satml2027ext/preflight_data_ext.py --data-root data_ext \
      --cifar-archive data/cifar-100-python.tar.gz --imagenette-archive data/imagenette2-320.tgz \
      --eurosat-archive data/eurosat/EuroSAT.zip --out results/satml2027ext/EXP021_DATA_PREFLIGHT.json
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import importlib
import platform
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from satml2027ext._common import PROJECT_ROOT, canonical_json_hash, read_json, sha256_file, write_json_atomic  # noqa: E402
from satml2027ext import datasets_ext, draw_items_ext  # noqa: E402
from satml2027ext.guard_ext import EXPERIMENT_IDS, load_registration  # noqa: E402

PREFLIGHT_SCHEMA = "satml2027ext.data_preflight.v1"
CIFAR100_MEMBER_MD5 = {
    "train": "16019d7e3df5f24257cddd939b257f8d",
    "test": datasets_ext.CIFAR100_TEST_MD5,
    "meta": datasets_ext.CIFAR100_META_MD5,
}
ENVIRONMENT_PACKAGES = ("numpy", "scipy", "torch", "torchvision", "open_clip", "huggingface_hub", "PIL", "safetensors")
RECORD_DISTRIBUTIONS = ("numpy", "scipy", "torch", "torchvision", "open_clip_torch", "huggingface_hub", "pillow", "safetensors")
FILE_DATASETS = ("eurosat", "imagenette")
EUROSAT_ARCHIVE_PREFIX = "eurosat/"  # item "eurosat/2750/<class>/<file>" is zip member "2750/<class>/<file>"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ENVIRONMENT_PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", None))
        except Exception:  # absent package: recorded, the worker compares
            versions[name] = None
    return versions


def distribution_records() -> dict[str, str | None]:
    """SHA-256 of each key distribution's installed RECORD file (it lists every installed file's hash)."""

    from importlib import metadata

    records: dict[str, str | None] = {}
    for name in RECORD_DISTRIBUTIONS:
        try:
            text = metadata.distribution(name).read_text("RECORD")
        except metadata.PackageNotFoundError:
            text = None
        records[name] = None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()
    return records


def installed_distributions() -> dict[str, str]:
    from importlib import metadata

    found: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            found[str(name).lower()] = str(distribution.version)
    return dict(sorted(found.items()))


def runtime_environment() -> dict[str, Any]:
    """Versions and device facts that the worker must reproduce."""

    environment: dict[str, Any] = {"packages": package_versions(), "platform": platform.platform(),
                                   "distribution_record_sha256": distribution_records(),
                                   "installed_distributions": installed_distributions()}
    try:
        import torch

        environment["cuda"] = torch.version.cuda
        environment["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        environment["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:
        environment["cuda"] = environment["cudnn"] = None
        environment["devices"] = []
    return environment


def comparable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    """The subset the worker must match exactly."""

    return {"packages": dict(environment.get("packages", {})), "cuda": environment.get("cuda"),
            "cudnn": environment.get("cudnn"),
            "distribution_record_sha256": dict(environment.get("distribution_record_sha256", {}))}


def registered_data_block(registrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    blocks = [registration.get("data_roots") for registration in registrations.values()]
    if any(not isinstance(block, Mapping) for block in blocks):
        raise RuntimeError("every registration must carry a data_roots block")
    if any(canonical_json_hash(block) != canonical_json_hash(blocks[0]) for block in blocks):
        raise RuntimeError("the three registrations disagree on data_roots")
    return dict(blocks[0])


def registered_item_rows(items_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {}
    for path in sorted(items_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows[path.name] = list(csv.DictReader(handle))
    return rows


def file_dataset_of(item_id: str) -> str | None:
    if item_id.startswith(datasets_ext.EUROSAT_FOLDER + "/"):
        return "eurosat"
    if item_id.startswith(datasets_ext.IMAGENETTE_FOLDER + "/"):
        return "imagenette"
    return None


def image_file_hashes(data_root: Path, rows: Mapping[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    """``{dataset: {item_id: sha256}}`` for every file-backed item any registered list names."""

    hashes: dict[str, dict[str, str]] = {name: {} for name in FILE_DATASETS}
    for listed in rows.values():
        for row in listed:
            dataset = file_dataset_of(row["item_id"])
            if dataset is None or row["item_id"] in hashes[dataset]:
                continue
            path = data_root / row["item_id"]
            if not path.is_file():
                raise RuntimeError(f"registered image is absent from the data root: {row['item_id']}")
            hashes[dataset][row["item_id"]] = sha256_file(path)
    return {name: dict(sorted(values.items())) for name, values in hashes.items()}


def _stream_sha256(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def archive_member_hashes(eurosat_archive: Path, imagenette_archive: Path,
                          wanted: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """SHA-256 of the archive member behind every registered file-backed item (streamed; nothing decoded)."""

    found: dict[str, dict[str, str]] = {name: {} for name in FILE_DATASETS}
    eurosat = {item[len(EUROSAT_ARCHIVE_PREFIX):]: item for item in wanted.get("eurosat", {})
               if item.startswith(EUROSAT_ARCHIVE_PREFIX)}
    if eurosat:
        with zipfile.ZipFile(eurosat_archive) as archive:
            names = set(archive.namelist())
            for member, item in sorted(eurosat.items()):
                if member in names:
                    with archive.open(member) as handle:
                        found["eurosat"][item] = _stream_sha256(handle)
    imagenette = set(wanted.get("imagenette", {}))
    if imagenette:
        with tarfile.open(imagenette_archive, "r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name in imagenette:
                    handle = archive.extractfile(member)
                    if handle is not None:
                        found["imagenette"][member.name] = _stream_sha256(handle)
    return {name: dict(sorted(values.items())) for name, values in found.items()}


def compare_with_archives(file_hashes: Mapping[str, Mapping[str, str]],
                          member_hashes: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for dataset in FILE_DATASETS:
        files = file_hashes.get(dataset, {})
        members = member_hashes.get(dataset, {})
        missing = sorted(set(files) - set(members))
        differing = sorted(item for item in set(files) & set(members) if files[item] != members[item])
        report[dataset] = {"files": len(files), "matched": len(files) - len(missing) - len(differing),
                           "absent_from_archive": missing[:10], "absent_count": len(missing),
                           "differing": differing[:10], "differing_count": len(differing)}
    return report


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    registrations = {}
    for path in args.registrations:
        registration = load_registration(path)
        registrations[registration["experiment_id"]] = registration
    if set(registrations) != set(EXPERIMENT_IDS):
        raise RuntimeError(f"the preflight needs all three registrations {EXPERIMENT_IDS}")
    data_block = registered_data_block(registrations)
    checks: dict[str, Any] = {}
    failures: list[str] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            failures.append(name)

    cifar = data_block["cifar100"]
    check("cifar100_archive_md5", md5_file(args.cifar_archive) == cifar["archive_md5"], md5_file(args.cifar_archive))
    check("cifar100_archive_sha256", sha256_file(args.cifar_archive) == cifar["archive_sha256"], sha256_file(args.cifar_archive))
    for member, expected in CIFAR100_MEMBER_MD5.items():
        path = args.data_root / datasets_ext.CIFAR100_FOLDER / member
        observed = md5_file(path) if path.is_file() else None
        check(f"cifar100_{member}_md5", observed == expected == cifar["members_md5"][f"cifar-100-python/{member}"], observed)
    imagenette = data_block["imagenette"]
    observed = sha256_file(args.imagenette_archive)
    check("imagenette_archive_sha256", observed == imagenette["archive_sha256"], observed)
    for split, count in datasets_ext.IMAGENETTE_OFFICIAL_COUNTS.items():
        try:
            listed = datasets_ext.imagenette_split_items(args.data_root, split, require_official=True)
            check(f"imagenette_{split}_official", len(listed) == count, len(listed))
        except Exception as error:
            check(f"imagenette_{split}_official", False, str(error))
    eurosat = data_block["eurosat"]
    observed = sha256_file(args.eurosat_archive)
    check("eurosat_archive_sha256", observed == eurosat["archive_sha256"], observed)

    # 3. byte-identical regeneration of every registered item list (its inputs must be on the host)
    created_at = data_block["item_lists"]["created_at"]
    draw_inputs = {"phase3_assignments": PROJECT_ROOT / draw_items_ext.DEFAULT_PHASE3,
                   "phase4_assignments": PROJECT_ROOT / draw_items_ext.DEFAULT_PHASE4,
                   "exp019_items": PROJECT_ROOT / draw_items_ext.DEFAULT_EXP019_ITEMS}
    absent_inputs = {name: str(value) for name, value in draw_inputs.items() if not Path(value).exists()}
    check("item_list_inputs_present", not absent_inputs, absent_inputs or "all present")
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary) / "items"
        draw_args = argparse.Namespace(
            phase3_assignments=PROJECT_ROOT / draw_items_ext.DEFAULT_PHASE3,
            phase4_assignments=PROJECT_ROOT / draw_items_ext.DEFAULT_PHASE4,
            exp019_items=PROJECT_ROOT / draw_items_ext.DEFAULT_EXP019_ITEMS,
            data_root=args.data_root, out=out, skip_unavailable=False, created_at=created_at,
            allow_synthetic=False, imagenette_archive=args.imagenette_archive,
        )
        manifest_hash = None
        if absent_inputs:
            check("item_lists_byte_identical", False, "not attempted: item-list inputs are absent")
        else:
            manifest = draw_items_ext.build(draw_args)
            registered = sorted(path.name for path in args.items.iterdir() if path.is_file())
            produced = sorted(path.name for path in out.iterdir() if path.is_file())
            identical = registered == produced and all(filecmp.cmp(out / name, args.items / name, shallow=False) for name in produced)
            check("item_lists_byte_identical", identical, {"registered": len(registered), "produced": len(produced)})
            manifest_hash = manifest["_manifest_sha256"]
    expected_manifest = {registration["provenance"]["item_manifest_sha256"] for registration in registrations.values()}
    check("item_manifest_hash_registered", expected_manifest == {manifest_hash}, manifest_hash)

    rows = registered_item_rows(args.items)
    list_hashes = {name: sha256_file(args.items / name) for name in sorted(rows)}
    try:
        file_hashes = image_file_hashes(args.data_root, rows)
        check("registered_image_files_present", True, {name: len(values) for name, values in file_hashes.items()})
    except RuntimeError as error:
        file_hashes = {}
        check("registered_image_files_present", False, str(error))
    member_hashes = archive_member_hashes(args.eurosat_archive, args.imagenette_archive, file_hashes)
    comparison = compare_with_archives(file_hashes, member_hashes)
    for dataset, detail in comparison.items():
        check(f"{dataset}_files_equal_archive_members",
              bool(file_hashes) and detail["files"] > 0 and detail["matched"] == detail["files"], detail)

    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "registrations": {key: value["registration_sha256"] for key, value in sorted(registrations.items())},
        "data_root": {"name": args.data_root.name},
        "archives": {
            "cifar100": {"file": args.cifar_archive.name, "bytes": args.cifar_archive.stat().st_size},
            "imagenette": {"file": args.imagenette_archive.name, "bytes": args.imagenette_archive.stat().st_size},
            "eurosat": {"file": args.eurosat_archive.name, "bytes": args.eurosat_archive.stat().st_size},
        },
        "checks": checks,
        "item_manifest_sha256": manifest_hash,
        "item_list_file_sha256": list_hashes,
        "image_file_sha256": file_hashes,
        "image_files_tied_to_archive_members": True,
        "environment": runtime_environment(),
        "evaluation_pixels_decoded": False,
    }


def load_preflight(path: str | Path, registrations: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Read a preflight record and require status PASS (and, if given, the same registration hashes)."""

    record = read_json(path)
    if not isinstance(record, Mapping) or record.get("schema_version") != PREFLIGHT_SCHEMA:
        raise RuntimeError(f"{path} is not an EXP-021 data preflight record")
    if record.get("status") != "PASS" or record.get("failures"):
        raise RuntimeError(f"data preflight did not pass: {record.get('failures')}")
    if registrations is not None and dict(record.get("registrations", {})) != dict(registrations):
        raise RuntimeError("data preflight was run against different registrations")
    return dict(record)


def verify_item_file(record: Mapping[str, Any], items_dir: Path, file_name: str) -> str:
    """The item-list CSV must hash to the value the preflight recorded."""

    expected = record["item_list_file_sha256"].get(file_name)
    observed = sha256_file(Path(items_dir) / file_name)
    if expected is None or observed != expected:
        raise RuntimeError(f"item list {file_name} differs from the preflight record")
    return observed


def verify_image_file(record: Mapping[str, Any], data_root: Path, item_id: str) -> None:
    """A file-backed image must have exactly the preflight hash (EuroSAT, Imagenette)."""

    dataset = file_dataset_of(item_id)
    if dataset is None:
        return
    expected = record["image_file_sha256"].get(dataset, {}).get(item_id)
    if expected is None or sha256_file(Path(data_root) / item_id) != expected:
        raise RuntimeError(f"image {item_id} differs from the preflight record")


def verify_environment(record: Mapping[str, Any]) -> None:
    """The runtime must equal the preflight environment (packages, CUDA, cuDNN)."""

    recorded = comparable_environment(record.get("environment", {}))
    current = comparable_environment(runtime_environment())
    if recorded != current:
        difference = {key: (recorded.get(key), current.get(key)) for key in recorded if recorded.get(key) != current.get(key)}
        raise RuntimeError(f"runtime environment differs from the data preflight: {difference}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cifar-archive", required=True, type=Path)
    parser.add_argument("--imagenette-archive", required=True, type=Path)
    parser.add_argument("--eurosat-archive", required=True, type=Path)
    parser.add_argument("--items", type=Path, default=PROJECT_ROOT / "results/satml2027ext/items")
    parser.add_argument("--registrations", nargs=3, type=Path, default=[
        PROJECT_ROOT / "configs/satml2027ext/exp-20260921-021a.json",
        PROJECT_ROOT / "configs/satml2027ext/exp-20260921-021b.json",
        PROJECT_ROOT / "configs/satml2027ext/exp-20260921-021c.json",
    ])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    record = run_preflight(args)
    write_json_atomic(args.out, record)
    print(f"data preflight {record['status']}: {args.out} sha256={sha256_file(args.out)}")
    for name, value in record["checks"].items():
        print(f"  {'PASS' if value['pass'] else 'FAIL'}  {name}")
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
