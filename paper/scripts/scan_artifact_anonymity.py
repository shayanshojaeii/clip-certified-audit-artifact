#!/usr/bin/env python3
"""Scan a directory or archive for identity-bearing strings before the anonymous artifact is released.

The SaTML 2027 artifact must be fully anonymized (call for papers, Open Science). The
reviews of reviewer bundle V4 (D-137) found user names, OneDrive and home-directory
paths, a private host address and host notes in the private review bundles; none of
that may reach the public artifact. This scanner flags:

* Windows user paths (``C:\\Users\\<name>``), OneDrive paths and Unix home paths;
* private IPv4 addresses (10/8, 172.16/12, 192.168/16) and e-mail addresses, except the
  public contact addresses in ``ALLOWED_EMAILS`` and any given with ``--allow-email``;
* every ``--token`` given on the command line (names, user ids, lab names). Tokens are
  never stored in a file, so the scanner itself carries no identity.

It fails closed (V5 review 1, D-142). Containers are recognized by content, not by file
name: ZIP (including ``.pt``, ``.pth``, ``.npz``, ``.docx``), tar and gzip are opened and
their member names and members are scanned recursively. Every other file is scanned as
text when it decodes as UTF-8, and otherwise through its embedded ASCII and UTF-16 strings.
A PDF is scanned through its text, its standard and custom metadata and its XMP metadata.
Anything the scanner cannot inspect is itself a finding: nesting deeper than ``MAX_DEPTH``,
an unreadable container, or a PDF without ``pdftotext``/``pdfinfo`` on the path.

Usage:
  python scripts/scan_artifact_anonymity.py <dir-or-file> [--token NAME ...] [--allow-email ADDR ...] [--report out.json]
Exit status 0 means no finding; 1 means findings (listed per file with the first lines).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

MAX_DEPTH = 6
# Public contact addresses that published material may quote (the SaTML program chairs' address appears in the
# reference-verification notes). Keep this list narrow; every other address is a finding.
ALLOWED_EMAILS = frozenset({"pcchairs@satml.org"})
GENERIC_PATTERNS = {
    "windows_user_path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"'`]+", re.IGNORECASE),
    "onedrive_path": re.compile(r"OneDrive", re.IGNORECASE),
    "unix_home_path": re.compile(r"/home/[A-Za-z0-9_.-]+"),
    "private_ipv4": re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
    "email_address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}
PRINTABLE_ASCII = re.compile(rb"[\x20-\x7e\t]{4,}")
PRINTABLE_UTF16 = re.compile(rb"(?:[\x20-\x7e\t]\x00){4,}")


def patterns(tokens: list[str]) -> dict[str, re.Pattern]:
    found = dict(GENERIC_PATTERNS)
    for index, token in enumerate(tokens):
        found[f"token_{index + 1}"] = re.compile(re.escape(token), re.IGNORECASE)
    return found


def record(findings: dict, label: str, kind: str, where: str) -> None:
    entry = findings.setdefault(label, {"count": 0, "kinds": {}, "first": []})
    entry["count"] += 1
    entry["kinds"][kind] = entry["kinds"].get(kind, 0) + 1
    if len(entry["first"]) < 3:
        entry["first"].append(f"{where}: {kind}")


def scan_text(label: str, text: str, compiled: dict[str, re.Pattern], findings: dict, allowed: frozenset[str]) -> None:
    for number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in compiled.items():
            matches = pattern.findall(line) if name == "email_address" else ([True] if pattern.search(line) else [])
            if name == "email_address":
                matches = [address for address in matches if address.lower() not in allowed]
            if matches:
                record(findings, label, name, str(number))


def embedded_strings(data: bytes) -> str:
    ascii_runs = [run.decode("ascii") for run in PRINTABLE_ASCII.findall(data)]
    utf16_runs = [run.decode("utf-16-le") for run in PRINTABLE_UTF16.findall(data)]
    return "\n".join(ascii_runs + utf16_runs)


def pdf_text(data: bytes) -> str | None:
    """Text, standard and custom metadata, and XMP metadata of a PDF; None when the tools are missing."""

    if not (shutil.which("pdftotext") and shutil.which("pdfinfo")):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.pdf"
        path.write_bytes(data)
        parts = [
            subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, errors="replace").stdout,
            subprocess.run(["pdfinfo", "-custom", str(path)], capture_output=True, text=True, errors="replace").stdout,
            subprocess.run(["pdfinfo", "-meta", str(path)], capture_output=True, text=True, errors="replace").stdout,
        ]
    return "\n".join(parts)


def scan_members(label: str, members: list[tuple[str, bytes | None]], compiled: dict[str, re.Pattern],
                 findings: dict, allowed: frozenset[str], depth: int) -> int:
    scanned = 0
    for name, content in members:
        scan_text(f"{label}!{name}::member-name", name, compiled, findings, allowed)
        if content is not None:
            scanned += scan_bytes(f"{label}!{name}", content, compiled, findings, allowed, depth + 1)
    return scanned


def zip_members(data: bytes) -> list[tuple[str, bytes | None]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return [(info.filename, None if info.is_dir() else archive.read(info)) for info in archive.infolist()]


def tar_members(data: bytes) -> list[tuple[str, bytes | None]]:
    members: list[tuple[str, bytes | None]] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for info in archive.getmembers():
            handle = archive.extractfile(info) if info.isfile() else None
            members.append((info.name, handle.read() if handle is not None else None))
            if info.issym() or info.islnk():
                members.append((f"{info.name}::link-target {info.linkname}", None))
    return members


def is_tar(data: bytes) -> bool:
    return len(data) >= 512 and data[257:262] == b"ustar"


def scan_bytes(label: str, data: bytes, compiled: dict[str, re.Pattern], findings: dict,
               allowed: frozenset[str] = ALLOWED_EMAILS, depth: int = 0) -> int:
    """Scan one file's bytes; returns the number of leaf files inspected."""

    if depth > MAX_DEPTH:
        record(findings, label, "nesting_too_deep", f"depth {depth}")
        return 1
    try:
        if data[:4] in (b"PK\x03\x04", b"PK\x05\x06") or zipfile.is_zipfile(io.BytesIO(data)):
            return scan_members(label, zip_members(data), compiled, findings, allowed, depth)
        if data[:2] == b"\x1f\x8b":
            return scan_bytes(f"{label}!gunzip", gzip.decompress(data), compiled, findings, allowed, depth + 1)
        if is_tar(data):
            return scan_members(label, tar_members(data), compiled, findings, allowed, depth)
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError, ValueError) as error:
        record(findings, label, "unreadable_container", type(error).__name__)
        scan_text(f"{label}::embedded-strings", embedded_strings(data), compiled, findings, allowed)
        return 1
    if data[:5] == b"%PDF-":
        text = pdf_text(data)
        if text is None:
            record(findings, label, "pdf_tools_unavailable", "pdftotext/pdfinfo")
            text = embedded_strings(data)
        scan_text(label, text, compiled, findings, allowed)
        return 1
    try:
        scan_text(label, data.decode("utf-8"), compiled, findings, allowed)
    except UnicodeDecodeError:
        scan_text(f"{label}::embedded-strings", embedded_strings(data), compiled, findings, allowed)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path)
    parser.add_argument("--token", action="append", default=[], help="identity string to flag (repeatable; not stored)")
    parser.add_argument("--allow-email", action="append", default=[],
                        help="a further public contact address that may appear (repeatable)")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    compiled = patterns(args.token)
    allowed = frozenset(ALLOWED_EMAILS | {address.lower() for address in args.allow_email})
    findings: dict = {}
    scanned = 0
    if args.target.is_file():
        scanned = scan_bytes(args.target.name, args.target.read_bytes(), compiled, findings, allowed)
    else:
        for path in sorted(p for p in args.target.rglob("*") if p.is_file() or p.is_symlink()):
            label = path.relative_to(args.target).as_posix()
            scan_text(label + "::path", label, compiled, findings, allowed)
            if path.is_symlink():
                record(findings, label, "symlink", "the release tree must not contain links")
                continue
            scanned += scan_bytes(label, path.read_bytes(), compiled, findings, allowed)
    summary = {"target": args.target.name, "files_scanned": scanned, "files_with_findings": len(findings),
               "allowed_emails": sorted(allowed), "findings_by_kind": {}, "files": findings}
    for entry in findings.values():
        for kind, count in entry["kinds"].items():
            summary["findings_by_kind"][kind] = summary["findings_by_kind"].get(kind, 0) + count
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("target", "files_scanned", "files_with_findings", "findings_by_kind")}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
