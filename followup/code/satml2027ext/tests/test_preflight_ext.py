"""Tests for the V3 data preflight additions (internal re-review of V2: M2a, n3).

  .venv/Scripts/python -m pytest satml2027ext/tests/test_preflight_ext.py -q
"""

from __future__ import annotations

import io
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from satml2027ext import preflight_data_ext as pre  # noqa: E402


def make_data(tmp_path: Path):
    root = tmp_path / "data"
    river = root / "eurosat" / "2750" / "River"
    river.mkdir(parents=True)
    (river / "River_1.jpg").write_bytes(b"river-1")
    (river / "River_2.jpg").write_bytes(b"river-2")
    tench = root / "imagenette2-320" / "val" / "n01440764"
    tench.mkdir(parents=True)
    (tench / "a.JPEG").write_bytes(b"tench-a")
    euro = tmp_path / "EuroSAT.zip"
    with zipfile.ZipFile(euro, "w") as archive:
        archive.writestr("2750/River/River_1.jpg", b"river-1")
        archive.writestr("2750/River/River_2.jpg", b"river-2")
    tgz = tmp_path / "imagenette2-320.tgz"
    with tarfile.open(tgz, "w:gz") as archive:
        data = b"tench-a"
        info = tarfile.TarInfo("imagenette2-320/val/n01440764/a.JPEG")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    rows = {"list.csv": [{"item_id": "eurosat/2750/River/River_1.jpg"}, {"item_id": "eurosat/2750/River/River_2.jpg"},
                         {"item_id": "imagenette2-320/val/n01440764/a.JPEG"}, {"item_id": "cifar100-test-00001"}]}
    return root, euro, tgz, rows


def test_v3_m2a_unpacked_files_are_tied_to_their_archive_members(tmp_path):
    root, euro, tgz, rows = make_data(tmp_path)
    files = pre.image_file_hashes(root, rows)
    report = pre.compare_with_archives(files, pre.archive_member_hashes(euro, tgz, files))
    assert (report["eurosat"]["matched"], report["imagenette"]["matched"]) == (2, 1)
    (root / "eurosat" / "2750" / "River" / "River_2.jpg").write_bytes(b"re-encoded")
    files = pre.image_file_hashes(root, rows)
    report = pre.compare_with_archives(files, pre.archive_member_hashes(euro, tgz, files))
    assert report["eurosat"]["differing_count"] == 1 and report["eurosat"]["differing"] == ["eurosat/2750/River/River_2.jpg"]
    extra = root / "imagenette2-320" / "val" / "n01440764" / "b.JPEG"
    extra.write_bytes(b"not in the archive")
    rows["list.csv"].append({"item_id": "imagenette2-320/val/n01440764/b.JPEG"})
    files = pre.image_file_hashes(root, rows)
    report = pre.compare_with_archives(files, pre.archive_member_hashes(euro, tgz, files))
    assert report["imagenette"]["absent_count"] == 1


def test_v3_n3_environment_binds_package_builds():
    environment = pre.runtime_environment()
    comparable = pre.comparable_environment(environment)
    assert set(comparable["distribution_record_sha256"]) == set(pre.RECORD_DISTRIBUTIONS)
    assert comparable["distribution_record_sha256"]["numpy"] is not None
    assert "numpy" in environment["installed_distributions"]
