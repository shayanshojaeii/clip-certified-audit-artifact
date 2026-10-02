# SaTML 2027 manuscript source (paper 579)

Unmodified `\documentclass[conference]{IEEEtran}`, default 10pt, two columns.
Body limit is 12 pages; the Open Science, LLM-usage and Ethical Considerations
sections, references and appendices do not count.

## Build

```bash
cd paper/satml2027
../../.venv/Scripts/python ../../scripts/generate_satml2027_paper_assets.py
latexmk -pdf main.tex
```

`latexmkrc` in this directory configures pdflatex + bibtex. The build fails
closed if `generated/macros.tex` is missing: every result number in the prose
and every table is a generated macro or file (see `MACROS_CONTRACT.md`).
`generated/sources.json` records the SHA-256 of every artifact that was read.

## Layout

- `main.tex` – document skeleton, theorem environments, macro import.
- `sections/abstract.tex` – the registered abstract, verbatim.
- `sections/introduction.tex`, `background.tex`, `operators.tex`,
  `audit_design.tex`, `results.tex`, `positive_control.tex`, `extension.tex`,
  `related_work.tex`, `limitations.tex`, `conclusion.tex`.
- `sections/theory_decision_change.tex` – per-draw identity, decision-change
  condition, Chowers invariance, flip budget (proofs in `appendix_proofs.tex`).
- `sections/open_science.tex`, `llm_usage.tex`, `ethics.tex` – required
  unnumbered sections, in this order, immediately before the references.
- `sections/appendix_nonidentification.tex`, `appendix_proofs.tex`,
  `appendix_tables.tex`.
- `generated/` – macros and tables produced by the generator; never edited by hand.
- `references.bib`, `references_extra.bib` – verified bibliography
  (`REFERENCES_VERIFICATION.md`); `main.tex` reads both.

## Rules

- Do not type a result value into any `.tex` file. Add a macro to the contract
  and the generator instead.
- Do not change the registered title or the substance of the registered abstract.
- The `EXTENSION-STATUS-BLOCK` comment in `sections/extension.tex` marks the
  only paragraph that may change depending on whether the registered extension
  is reconstructed and independently reviewed before submission. Do not inspect
  partial outcomes to decide.
- Anonymity: no author names, affiliations, or identifying links anywhere,
  including PDF metadata.
