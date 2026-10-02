"""Phrase and number lint for the SaTML 2027 manuscript sections.

Fails if any section contains a prohibited claim phrase, a hand-typed result
number (a decimal with a percent sign or a signed decimal outside a macro), an
identifying string, or a reference to a label that no file defines.

Usage: .venv/Scripts/python scripts/lint_satml2027_claims.py [--token NAME ...]

Identity strings (author names, user ids, places) are not stored in this file, so
the released script carries no identity (D-137; the reviews of reviewer bundle V4
asked for a scrubbed artifact). Pass them at run time with ``--token`` (repeatable)
or in the environment variable ``SATML2027_ANONYMITY_TOKENS`` (separated by ``;``).
Generic identity patterns (user and home paths, OneDrive, private IPv4 addresses,
e-mail addresses) are always checked.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "satml2027"
SECTIONS = sorted((PAPER / "sections").glob("*.tex"))

# (pattern, reason, exempt-if-line-matches)
NEGATED = r"\b(not|no|never|neither|nor|rather than|descriptiv\w*|identif\w*|make no|makes no|cannot|without)\b"
PROHIBITED = [
    (r"\bequivalen(t|ce)\b", "equivalence claim", NEGATED),
    (r"\bno (meaningful|real) effect\b", "evidence-of-absence claim", None),
    (r"\bindistinguishable\b", "evidence-of-absence wording", None),
    (r"\bpredicts? (what the|the table|the outcome|the result|the audit)\b", "hindsight theory framed as prediction", None),
    (r"\bidentical (to|on) .* every outcome\b", "per-item overstatement", None),
    (r"\bprov(e|ed|es) (that )?(no|there is no) effect\b", "evidence-of-absence claim", None),
    (r"\bcaus(e|es|ed|al)\b", "causal verb", NEGATED),
    (r"\bhub[- ]collapse\b", "causal mechanism label", None),
    (r"\battractor collapse\b", "causal mechanism label", None),
    (r"\buniversal(ly)?\b", "universal-failure wording", NEGATED),
    (r"\bfails? (in general|universally|for all)\b", "universal-failure wording", None),
    (r"\bfirst (certified|to certify)\b", "priority claim", None),
    (r"\bnovel\b", "novelty adjective", None),
    (r"\bstate[- ]of[- ]the[- ]art\b", "marketing", None),
    (r"\bD\^?\\?\*|\\tau\^?\\?\*|tau\\_\\?star", "FSS symbol in Cohen context", None),
    (r"\banchored certified accuracy\b[^.]*\bstandard\b[^.]*\bsame\b", "metric conflation", None),
    # Wording corrected after the three reviews of reviewer bundle V5 (2026-09-27); kept out for good.
    (r"\bcensored alike\b", "paired-censoring symmetry claim", None),
    (r"\bevery\}?\s+clean\s+statistic", "non-identification overstatement", None),
    (r"\btoo small to act\b", "bound stated as an observation", None),
    (r"\+\s*\$?\\GR(Std|Anch)Delta", "duplicated sign: the macro carries its sign", None),
]

# File-scoped exemptions (D-145): the EXP-021B registration makes "practically equivalent" a registered outcome
# class (Q6: Holm-adjusted two one-sided tests against the 2-point SESOI), so only the follow-up sections may name
# that class. The audit and the extension register no equivalence test and stay under the rule above.
REGISTERED_CLASS_FILES = {"followup.tex", "appendix_followup.tex"}
REGISTERED_CLASS = r"\bpractically equivalent\b"

GENERIC_IDENTIFYING = [
    r"[A-Za-z]:[\\/]+Users[\\/]+",
    r"/home/[A-Za-z0-9_.-]+",
    r"OneDrive",
    r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    r"vast\.ai",
]


def prohibited_reasons(file_name: str, stripped: str) -> list[str]:
    """The prohibited-phrase reasons of one comment-stripped line of the section ``file_name``."""

    reasons = []
    for pattern, why, exempt in PROHIBITED:
        if not re.search(pattern, stripped, flags=re.IGNORECASE):
            continue
        if exempt and re.search(exempt, stripped, flags=re.IGNORECASE):
            continue
        if (why == "equivalence claim" and file_name in REGISTERED_CLASS_FILES
                and not re.search(pattern, re.sub(REGISTERED_CLASS, "", stripped, flags=re.IGNORECASE), flags=re.IGNORECASE)):
            continue  # the only equivalence wording on the line is the registered class name
        reasons.append(why)
    return reasons


def identifying_patterns(tokens: list[str]) -> list[str]:
    """Generic patterns plus the run-time identity tokens (never stored in this file)."""

    extra = [token for token in os.environ.get("SATML2027_ANONYMITY_TOKENS", "").split(";") if token.strip()]
    return GENERIC_IDENTIFYING + [re.escape(token.strip()) for token in [*tokens, *extra] if token.strip()]

# Hand-typed numbers: percentages or signed decimals with two decimals not inside a macro.
TYPED_NUMBER = re.compile(r"(?<![\\\w{])[+-]?\d+\.\d{2,}(?=\s*(\\%|%|\s|,|\)|\]))")
ALLOWED_NUMBER_CONTEXT = re.compile(
    r"(sigma|\\sigma|r=|radius|step|coefficient|c=|\\alpha|alpha|compat|width|height"
    r"|Python|PyTorch|CUDA|OpenCLIP|version|0\.25|0\.12|0\.5\b|0\.1\b|0\.0025|0\.005|0\.01|0\.02|0\.04)"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--token", action="append", default=[], help="identity string to flag (repeatable; not stored)")
    args = parser.parse_args(argv)
    identifying = identifying_patterns(args.token)
    problems: list[str] = []
    defined_labels: set[str] = set()
    referenced_labels: set[str] = set()
    for path in list(SECTIONS) + sorted((PAPER / "generated").glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        defined_labels.update(re.findall(r"\\label\{([^}]+)\}", text))
        referenced_labels.update(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", text))
        if path.parent.name == "generated":
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.split("%", 1)[0]
            for why in prohibited_reasons(path.name, stripped):
                problems.append(f"{path.relative_to(ROOT)}:{line_number}: {why}: {stripped.strip()[:110]}")
            for pattern in identifying:
                if re.search(pattern, stripped, flags=re.IGNORECASE):
                    problems.append(f"{path.relative_to(ROOT)}:{line_number}: identifying string: {stripped.strip()[:110]}")
            for match in TYPED_NUMBER.finditer(stripped):
                window = stripped[max(0, match.start() - 40): match.end() + 10]
                if ALLOWED_NUMBER_CONTEXT.search(window):
                    continue
                problems.append(f"{path.relative_to(ROOT)}:{line_number}: hand-typed number '{match.group(0)}': {stripped.strip()[:110]}")
    missing = sorted(referenced_labels - defined_labels)
    for label in missing:
        problems.append(f"undefined label referenced: {label}")
    if problems:
        print("\n".join(problems))
        print(f"\n{len(problems)} problem(s)")
        return 1
    print("lint clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
