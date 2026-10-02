"""Source-completeness checks for the SaTML 2027 manuscript (review of reviewer bundle V4, M-A1).

A clean extraction of manuscript review V3 lacked ``references_extra.bib``, although
``main.tex`` names it, and five citations were left undefined; the already-built PDF hid
the gap. These checks run on the source tree alone (no TeX needed), so they also fail
inside a review package that omits a file.

  .venv/Scripts/python -m pytest tests/test_satml2027_manuscript_sources.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "satml2027"
COMMENT = re.compile(r"(?<!\\)%.*")


def _source(path: Path) -> str:
    return "\n".join(COMMENT.sub("", line) for line in path.read_text(encoding="utf-8").splitlines())


def _inputs(path: Path, seen: set[Path]) -> list[Path]:
    """``path`` and every file it reaches through \\input, depth first."""

    seen.add(path)
    reached = [path]
    for target in re.findall(r"\\input\{([^}]+)\}", _source(path)):
        child = PAPER / (target if target.endswith(".tex") else f"{target}.tex")
        assert child.is_file(), f"{path.relative_to(PAPER)} inputs a missing file: {target}"
        if child not in seen:
            reached += _inputs(child, seen)
    return reached


def _bib_files() -> list[Path]:
    names = re.findall(r"\\bibliography\{([^}]+)\}", _source(PAPER / "main.tex"))
    assert len(names) == 1, "main.tex must call \\bibliography exactly once"
    return [PAPER / f"{name.strip()}.bib" for name in names[0].split(",")]


def _bib_keys(path: Path) -> list[str]:
    return re.findall(r"^\s*@\w+\s*\{\s*([^,\s]+)\s*,", path.read_text(encoding="utf-8"), flags=re.MULTILINE)


def test_every_bibliography_file_named_by_main_tex_exists_and_has_entries():
    files = _bib_files()
    assert [path.name for path in files] == ["references.bib", "references_extra.bib"]
    for path in files:
        assert path.is_file(), f"main.tex names {path.name}, which is missing"
        assert _bib_keys(path), f"{path.name} defines no entry"


def test_bibliography_keys_are_unique_across_files():
    keys = [key for path in _bib_files() for key in _bib_keys(path)]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    assert not duplicates, f"keys defined twice: {duplicates}"


def test_every_citation_in_the_manuscript_is_defined():
    defined = {key for path in _bib_files() for key in _bib_keys(path)}
    cited: set[str] = set()
    for path in _inputs(PAPER / "main.tex", set()):
        for group in re.findall(r"\\(?:no)?cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", _source(path)):
            cited.update(key.strip() for key in group.split(",") if key.strip())
    missing = sorted(cited - defined)
    assert not missing, f"cited but not defined in the bibliography files: {missing}"
    assert {"westfall1993resampling", "mohapatra2021hidden", "anani2024adaptive", "card2020power",
            "bertinetto2021preregistration"} <= cited  # the five keys of references_extra.bib are live citations


def test_references_verification_covers_every_bibliography_entry():
    log = (PAPER / "REFERENCES_VERIFICATION.md").read_text(encoding="utf-8")
    unlogged = sorted(key for path in _bib_files() for key in _bib_keys(path) if f"`{key}`" not in log)
    assert not unlogged, f"entries without a verification record: {unlogged}"


def test_claim_lint_carries_no_identity_and_flags_generic_patterns_and_runtime_tokens():
    """D-137: identity strings are passed at run time; the released lint stores only generic patterns."""

    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location("lint_claims_under_test", ROOT / "scripts" / "lint_satml2027_claims.py")
    lint = importlib.util.module_from_spec(spec)
    _sys.modules["lint_claims_under_test"] = lint
    spec.loader.exec_module(lint)
    patterns = lint.identifying_patterns(["Example-Surname"])

    def flagged(line: str) -> bool:
        return any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns)

    for line in ("C:" + "\\" + "Users" + "\\" + "someone" + "\\" + "data", "/home/someone/data", "host 172.16.3.4",
                 "a@b.example", "OneDrive folder", "by example-surname"):
        assert flagged(line), line
    for line in ("radius 0.25 and 3.14 points", "Section IV", "the 172.15.0.1 note"):
        assert not flagged(line), line


def test_claim_lint_allows_only_the_registered_equivalence_class_in_the_follow_up():
    """D-145: EXP-021B registers "practically equivalent" (Holm-adjusted TOST); only the follow-up files may name it."""

    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location("lint_claims_exemption", ROOT / "scripts" / "lint_satml2027_claims.py")
    lint = importlib.util.module_from_spec(spec)
    _sys.modules["lint_claims_exemption"] = lint
    spec.loader.exec_module(lint)
    assert lint.prohibited_reasons("followup.tex", "3 practically equivalent and 2 inconclusive") == []
    assert lint.prohibited_reasons("appendix_followup.tex", "class practically equivalent") == []
    assert lint.prohibited_reasons("followup.tex", "so the two banks are equivalent") == ["equivalence claim"]
    assert lint.prohibited_reasons("followup.tex", "practically equivalent, hence equivalence") == ["equivalence claim"]
    assert lint.prohibited_reasons("extension.tex", "3 practically equivalent") == ["equivalence claim"]
    assert lint.prohibited_reasons("extension.tex", "no equivalence is claimed") == []
