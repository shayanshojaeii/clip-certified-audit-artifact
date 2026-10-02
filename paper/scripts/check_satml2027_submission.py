#!/usr/bin/env python3
"""Submission gate for the SaTML 2027 PDF (D-149): refuses a PDF that is not ready to upload.

Checks, each printed as PASS or FAIL (exit status 1 on any FAIL):
* the PDF and its build log are newer than every section, generated asset and bibliography file;
* the page-limited body ends on or before page 12 (the page of \\label{body:end} in main.aux);
* the build log has no undefined reference or citation and no LaTeX error;
* no generated macro is missing (generated/sources.json "MISSING" is empty), the PDF text has no "??" and no
  figure placeholder, and no section still carries a "to be written" marker;
* every font is embedded (pdffonts), and the PDF author metadata is "Anonymous";
* the PDF text carries no identifying string (the claim lint's generic patterns plus run-time tokens passed with
  --token or SATML2027_ANONYMITY_TOKENS; tokens are never stored in this file).
Overfull boxes wider than 5pt are reported as warnings.

Usage (repository root, after latexmk): .venv/Scripts/python scripts/check_satml2027_submission.py [--token NAME ...]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "satml2027"
PAGE_LIMIT = 12
sys.path.insert(0, str(ROOT / "scripts"))


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--token", action="append", default=[], help="identity string to flag (repeatable; not stored)")
    args = parser.parse_args(argv)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    pdf, log, aux = PAPER / "main.pdf", PAPER / "main.log", PAPER / "main.aux"
    for path in (pdf, log, aux):
        check(f"{path.name} exists", path.is_file())
    if not all(path.is_file() for path in (pdf, log, aux)):
        return report(results)
    inputs = [*PAPER.glob("sections/*.tex"), *PAPER.glob("generated/*"), *PAPER.glob("*.bib"), PAPER / "main.tex"]
    stale = sorted(path.name for path in inputs if path.stat().st_mtime > pdf.stat().st_mtime)
    check("the PDF is newer than every source", not stale, ", ".join(stale[:5]))

    match = re.search(r"\\newlabel\{body:end\}\{\{[^}]*\}\{(\d+)\}", aux.read_text(encoding="utf-8", errors="replace"))
    page = int(match.group(1)) if match else None
    check(f"the body ends on or before page {PAGE_LIMIT}", page is not None and page <= PAGE_LIMIT,
          f"body:end on page {page}")

    log_text = log.read_text(encoding="utf-8", errors="replace")
    undefined = re.findall(r"(?:Reference|Citation) `[^']+' on page \d+ undefined", log_text)
    check("no undefined reference or citation", not undefined and "There were undefined" not in log_text,
          "; ".join(undefined[:3]))
    check("no LaTeX error in the log", not re.search(r"^! ", log_text, flags=re.MULTILINE))
    overfull = [float(value) for value in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log_text)]
    wide = [value for value in overfull if value > 5.0]

    sources = json.loads((PAPER / "generated" / "sources.json").read_text(encoding="utf-8"))
    missing = [entry["macro"] for entry in sources.get("MISSING", [])]
    check("no generated macro is missing", not missing, ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else ""))
    text = run(["pdftotext", "-layout", str(pdf), "-"])
    check("the PDF text has no '??'", "??" not in text, f"{text.count('??')} occurrence(s)")
    check("the PDF text has no figure placeholder", "Placeholder:" not in text, f"{text.count('Placeholder:')} placeholder(s)")
    markers = [path.name for path in PAPER.glob("sections/*.tex")
               if re.search(r"to be written", path.read_text(encoding="utf-8"), flags=re.IGNORECASE)]
    check("no section carries a 'to be written' marker", not markers, ", ".join(markers))

    fonts = run(["pdffonts", str(pdf)]).splitlines()[2:]
    # columns: name, type (may contain spaces), encoding, emb, sub, uni, object ID (two numbers): emb is 5th from the right
    unembedded = [line.split()[0] for line in fonts if len(line.split()) >= 7 and line.split()[-5] != "yes"]
    check("every font is embedded", not unembedded, ", ".join(unembedded[:5]))
    info = run(["pdfinfo", str(pdf)])
    author = re.search(r"^Author:\s*(.*)$", info, flags=re.MULTILINE)
    check("PDF author metadata is 'Anonymous'", author is not None and author.group(1).strip() == "Anonymous",
          author.group(1).strip() if author else "no Author field")

    import lint_satml2027_claims as lint

    identifying = [pattern for pattern in lint.identifying_patterns(args.token) if re.search(pattern, text, flags=re.IGNORECASE)]
    check("no identifying string in the PDF text", not identifying, f"{len(identifying)} pattern(s) matched")
    pages = re.search(r"^Pages:\s*(\d+)", info, flags=re.MULTILINE)
    print(f"PDF: {pdf.relative_to(ROOT)}, {pages.group(1) if pages else '?'} pages; body ends on page {page}")
    if wide:
        print(f"warning: {len(wide)} overfull box(es) wider than 5pt (max {max(wide):.1f}pt)")
    return report(results)


def report(results: list[tuple[str, bool, str]]) -> int:
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail and not ok else ""))
    failed = sum(1 for _, ok, _ in results if not ok)
    print("SUBMISSION GATE: " + ("PASS" if not failed else f"FAIL ({failed})"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
