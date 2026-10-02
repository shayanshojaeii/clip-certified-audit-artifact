# Reproducing the paper's numbers on a CPU

Requirements: Python 3.12 with NumPy, SciPy, pytest and CPU PyTorch (PyTorch reads the audit's saved `.pt`
statistics). Building the PDF needs a TeX distribution with `latexmk`, the IEEEtran class and BibTeX.

## 1. Stage the files

The folders are organized by study. The code reads its inputs at the paths it was written for, so first copy every
released file into that layout (about the size of the repository):

```bash
python paper/stage_inputs.py run
cd run
```

`stage_inputs.py` checks each file against `MANIFEST.sha256.json` before copying it.

## 2. Recompute

```bash
python scripts/exp021_replay_results.py                # follow-ups A-C: every registered output from the vote counts,
                                                       # the custody and approval bindings, the radius curves, the macros
python scripts/exp021_replay_results.py --macros-only  # every paper macro regenerates identically from the released outputs
python scripts/generate_satml2027_figure_data.py       # figure data of the complete-family band
python scripts/exp021_exact_boundary_recompute.py exact  # exact-integer bootstrap tails beside the registered ones
                                                       # (Appendix E): compare exact/ with analysis/exp021_exact_boundary/
python scripts/lint_satml2027_claims.py                # claim lint of the manuscript
python -m pytest -q tests/test_generate_satml2027_paper_assets.py tests/test_followup_assets.py tests/test_satml2027_manuscript_sources.py tests/test_decision_change_condition.py
cd paper/satml2027 && latexmk -pdf main.tex            # the submitted PDF from source
```

The follow-up approvals are released with the approver's name replaced by a role. Their original SHA-256, which the
run custody manifests bind, are listed in `followup/approvals/REDACTION.json`; the replay checks each redacted copy
against that record. The unredacted files are released with the final version.

## 3. Further checks

- The extension's registered analysis outputs are in `extension/registered_analysis/`; the reconstruction package in
  `extension/reconstruction/` recomputes them from the vote counts in `extension/vote_counts/`.
- The positive control's custody package (`control/custody/`) recomputes its aggregates and paired bootstrap from
  the confirmation outputs in `control/run/`.
- `analysis/scripts/` recompute the metric-semantics audit, the band reconstruction and the exact-count adjudication
  from `results/exp017/`.
