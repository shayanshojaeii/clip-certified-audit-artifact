"""Tests of the fail-closed anonymity scanner (V5 review 1, D-142)."""

from __future__ import annotations

import gzip
import io
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scan_artifact_anonymity as scanner  # noqa: E402


def run(target: Path, *extra: str) -> tuple[int, dict]:
    report = target.parent / f"{target.name}.report.json"
    status = scanner.main([str(target), "--report", str(report), *extra])
    return status, json.loads(report.read_text(encoding="utf-8"))


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def tree(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "release"
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_clean_tree_passes(tmp_path):
    root = tree(tmp_path, {"README.md": b"Anonymous artifact.\n", "data/values.csv": b"a,b\n1,2\n"})
    status, report = run(root)
    assert status == 0 and report["files_with_findings"] == 0 and report["files_scanned"] == 2


def test_review_probe_pt_zip_with_windows_path_is_caught(tmp_path):
    # Review 1 of bundle V5: a .pt file that is a ZIP container with a path in a text member passed the old scanner.
    probe = zip_bytes({"archive/data.pkl": b"\x80\x02}q\x00.", "archive/notes.txt": b"C:\\Users\\reviewer\\secret\n"})
    root = tree(tmp_path, {"model/weights.pt": probe})
    status, report = run(root)
    assert status == 1
    assert report["findings_by_kind"].get("windows_user_path") == 1
    assert "model/weights.pt!archive/notes.txt" in report["files"]


@pytest.mark.parametrize("suffix", [".pth", ".npz", ".bin", ""])
def test_zip_detected_by_content_for_any_suffix(tmp_path, suffix):
    root = tree(tmp_path, {f"blob{suffix}": zip_bytes({"x/y.npy": b"\x93NUMPY /home/alice/run\n"})})
    status, report = run(root)
    assert status == 1 and report["findings_by_kind"].get("unix_home_path", 0) >= 1


def test_member_names_are_scanned(tmp_path):
    root = tree(tmp_path, {"arrays.npz": zip_bytes({"OneDrive/Desktop/a.npy": b"\x93NUMPY"})})
    status, report = run(root)
    assert status == 1 and "arrays.npz!OneDrive/Desktop/a.npy::member-name" in report["files"]


def test_binary_file_scanned_through_embedded_ascii_and_utf16_strings(tmp_path):
    ascii_blob = b"\x00\xff\x10" + b"host 192.168.1.20 used" + b"\xfe\x00"
    utf16_blob = b"\xff\xfe\x00" + "C:\\Users\\bob\\x".encode("utf-16-le") + b"\x00\x81"
    root = tree(tmp_path, {"a.bin": ascii_blob, "b.dat": utf16_blob})
    status, report = run(root)
    assert status == 1
    assert report["findings_by_kind"].get("private_ipv4") == 1
    assert report["findings_by_kind"].get("windows_user_path") == 1


def test_tar_gz_members_are_scanned(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        content = b"server 10.0.0.7\n"
        info = tarfile.TarInfo("logs/run.log")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    root = tree(tmp_path, {"results.tgz": buffer.getvalue()})
    status, report = run(root)
    assert status == 1 and report["findings_by_kind"].get("private_ipv4") == 1


def test_gzip_payload_is_scanned(tmp_path):
    root = tree(tmp_path, {"notes.gz": gzip.compress(b"contact someone@example.org\n")})
    status, report = run(root)
    assert status == 1 and report["findings_by_kind"].get("email_address") == 1


def test_nesting_beyond_the_limit_is_a_finding(tmp_path):
    payload = b"clean text\n"
    for level in range(scanner.MAX_DEPTH + 2):
        payload = zip_bytes({f"level{level}.zip": payload})
    root = tree(tmp_path, {"deep.zip": payload})
    status, report = run(root)
    assert status == 1 and report["findings_by_kind"].get("nesting_too_deep") == 1


def test_unreadable_container_is_a_finding(tmp_path):
    corrupt = zip_bytes({"a.txt": b"hello"})[:30]
    root = tree(tmp_path, {"broken.pt": corrupt})
    status, report = run(root)
    assert status == 1 and report["findings_by_kind"].get("unreadable_container") == 1


def test_public_contact_allowed_other_addresses_flagged(tmp_path):
    root = tree(tmp_path, {"REFS.md": b"Asked pcchairs@satml.org (public contact).\n"})
    assert run(root)[0] == 0
    root2 = tree(tmp_path / "second", {"notes.md": b"write to author@university.edu\n"})
    status, report = run(root2)
    assert status == 1 and report["findings_by_kind"].get("email_address") == 1
    assert run(root2, "--allow-email", "author@university.edu")[0] == 0


def test_tokens_are_flagged_and_never_stored(tmp_path):
    root = tree(tmp_path, {"a.md": b"Prepared by Jane Q. Researcher\n"})
    status, report = run(root, "--token", "Researcher")
    assert status == 1 and report["findings_by_kind"].get("token_1") == 1
    assert "Researcher" not in (ROOT / "scripts" / "scan_artifact_anonymity.py").read_text(encoding="utf-8")


def _minimal_pdf(info_entries: str) -> bytes:
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
        f"<< {info_entries} >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 4 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))
    return out.getvalue()


@pytest.mark.skipif(not (shutil.which("pdftotext") and shutil.which("pdfinfo")), reason="poppler tools not installed")
def test_pdf_custom_metadata_is_scanned(tmp_path):
    clean = _minimal_pdf("/Title (Anonymous) /Author (Anonymous)")
    leaky = _minimal_pdf("/Title (Anonymous) /SourcePath (C:\\\\Users\\\\carol\\\\paper.tex)")
    assert run(tree(tmp_path / "a", {"paper.pdf": clean}))[0] == 0
    status, report = run(tree(tmp_path / "b", {"paper.pdf": leaky}))
    assert status == 1 and report["findings_by_kind"].get("windows_user_path") == 1


def test_pdf_without_tools_is_a_finding(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner.shutil, "which", lambda name: None)
    status, report = run(tree(tmp_path, {"paper.pdf": _minimal_pdf("/Title (Anonymous)")}))
    assert status == 1 and report["findings_by_kind"].get("pdf_tools_unavailable") == 1
