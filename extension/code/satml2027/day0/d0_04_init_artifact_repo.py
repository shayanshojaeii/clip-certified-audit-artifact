#!/usr/bin/env python3
"""Day 0, step 4 - build the anonymized artifact repository skeleton.

SaTML requires an Open Science section and a fully anonymized artifact within
three days of the submission deadline, unedited thereafter.  This script lays
out the repository, copies the analysis code, writes REPRODUCE.md and an
anonymization checklist, and runs a scan for identifying strings.

Usage:
  python day0/d0_04_init_artifact_repo.py --out artifact --bundle . \
      [--include-results results/satml2027] [--scan-only]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hashing import write_json_atomic  # noqa: E402

LAYOUT = ("registrations", "code", "sufficient_statistics", "banks", "analysis", "figures", "logs")

IDENTIFYING_PATTERNS = (
    r"[\w.+-]+@[\w-]+\.[\w.]+",
    r"github\.com/[\w.-]+",
    r"/home/[\w.-]+",
    r"/Users/[\w.-]+",
    r"[A-Za-z]:\\Users\\[\w.-]+",
    r"\borcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b",
)

REPRODUCE = """# Reproducing every number in the paper

All tables and figures are regenerated from the released sufficient statistics;
no GPU is required for the analysis.

```bash
python -m pip install -r code/requirements.txt          # numpy, scipy only
python code/day1/d1_07_analyze.py \
  --merged sufficient_statistics/merged/EXP-20260920-019A \
  --registration registrations/exp-20260920-019a.json \
  --out analysis/EXP-20260920-019A
python code/day1/d1_07_analyze.py \
  --merged sufficient_statistics/merged/EXP-20260920-019B \
  --registration registrations/exp-20260920-019b.json \
  --out analysis/EXP-20260920-019B
```

Contents
--------
* `registrations/` - the preregistration files with their canonical-JSON SHA-256
  sidecars.  Each experiment's outputs carry the hash of the registration they
  were produced under.
* `sufficient_statistics/` - per cell: selection and confirmation vote counts for
  every bank `[banks, items, classes]`, raw predictions, ground truth, item
  identifiers and the per-item noise seeds.  These are the complete inputs to
  every reported number.
* `banks/` - every candidate and control prototype bank with its tensor hash.
* `code/` - the worker, the bank builders and the torch-free analysis scripts.
* `analysis/` - regenerated tables (written by the command above).

Custody
-------
`registrations/phase4_custody_report.json` records the hash chain from the
Phase-3 split table through the Phase-4 configuration to every output file, and
the item-role ledger records which items each experiment consumed.  The sealed
final-test split is not part of this release and was never read; every artifact
carries `final_test_access: false`.
"""

CHECKLIST = """# Anonymization checklist (run before submission)

- [ ] no author names, affiliations, acknowledgements or grant numbers in the PDF
- [ ] own prior work cited in the third person ("the authors' preregistration", not "our earlier EXP-017")
- [ ] PDF metadata scrubbed (`exiftool -all= paper.pdf` or `pdftk ... dump_data`)
- [ ] artifact repository contains no usernames, e-mail addresses, absolute home
      paths, institution names, cluster hostnames or ORCIDs (`d0_04 --scan-only`)
- [ ] repository is served from an anonymizing host (for example anonymous.4open.science)
- [ ] the repository is frozen after the artifact deadline and not edited during review
- [ ] Open Science, LLM usage considerations and Ethics sections are present and
      placed before the references, where they do not count towards the page limit
"""


def scan(root: Path) -> list[dict[str, str]]:
    findings = []
    patterns = [re.compile(pattern) for pattern in IDENTIFYING_PATTERNS]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix in {".pt", ".npz", ".png", ".pdf", ".jpg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in patterns:
            for match in pattern.findall(text):
                findings.append({"path": path.as_posix(), "pattern": pattern.pattern, "match": str(match)})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-results", type=Path, default=None)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    if args.scan_only:
        findings = scan(args.out)
        print(f"{len(findings)} potential identifying strings")
        for finding in findings[:50]:
            print(f"  {finding['path']}: {finding['match']}")
        return 1 if findings else 0

    for name in LAYOUT:
        (args.out / name).mkdir(parents=True, exist_ok=True)
    code = args.out / "code"
    for sub in ("common", "day0", "day1"):
        source = args.bundle / sub
        if source.is_dir():
            shutil.copytree(source, code / sub, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (code / "requirements.txt").write_text("numpy>=1.24\nscipy>=1.10\n", encoding="utf-8")
    (args.out / "REPRODUCE.md").write_text(REPRODUCE, encoding="utf-8")
    (args.out / "ANONYMIZATION_CHECKLIST.md").write_text(CHECKLIST, encoding="utf-8")
    if args.include_results and args.include_results.is_dir():
        shutil.copytree(args.include_results, args.out / "sufficient_statistics", dirs_exist_ok=True)
    write_json_atomic(
        args.out / "artifact_info.json",
        {
            "schema_version": "satml2027.artifact_info.v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "final_test_access": False,
            "contents": list(LAYOUT),
            "note": "anonymous artifact for double-blind review; frozen after the artifact deadline",
        },
    )
    findings = scan(args.out)
    print(f"artifact skeleton at {args.out}")
    print(f"anonymization scan: {len(findings)} potential identifying strings"
          + (" (review them before publishing)" if findings else ""))
    for finding in findings[:20]:
        print(f"  {finding['path']}: {finding['match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
