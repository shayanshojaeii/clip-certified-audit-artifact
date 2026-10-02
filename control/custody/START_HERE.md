# START HERE: independent result-custody review of EXP-20260917-020-FARLA-FULL

Package: `FARLA_FULL_RESULT_CUSTODY_V1` (prepared 2026-09-25, Person A).
Reviewer: the other team member (Person B), working independently.
Repository location: `FARLA/artifacts/phase4/FARLA_FULL_RESULT_CUSTODY_V1/`.

Nothing in this package changes a result. It exists so that you can sign off
on the completed FARLA pilot without trusting FARLA's own analysis code, or
refuse to. Claims-matrix row C10 already states that final paper use of these
numbers requires this sign-off.

## 1. What you are asked to decide

Whether the numbers below, taken from the frozen `analysis/report.json` of
`FARLA/downloaded_results/EXP-20260917-020-FARLA-FULL/`, are (a) computed from
the result files whose digests are in `manifest.sha256.json`, (b) reproducible
by an independent reconstruction from the raw stored vote counts, and (c)
derived from data roles that are disjoint from each other and from every
Phase-3 EXP-017 development, validation, excluded, and sealed-test item.

Macro standard Cohen certified accuracy at radius 0.25 (sigma 0.25, 4096 draws,
alpha 0.001, equal-cell mean over the four model x dataset cells), with macro
clean accuracy:

| Candidate | Std CA@0.25 | Clean acc | Anchored CA@0.25 |
|---|---:|---:|---:|
| no_correction (frozen zero-shot prototypes) | 0.1160 | 0.6088 | 0.0843 |
| shared_translation_ce_supervised | 0.1175 | 0.6128 | 0.0868 |
| lowrank_ce_r8_supervised (matched noisy-CE control) | 0.2710 | 0.7250 | 0.2335 |
| lowrank_tail_r8_supervised (tail-only control) | 0.3650 | 0.7080 | 0.2875 |
| farla_full_r8_supervised (primary candidate) | 0.3505 | 0.6858 | 0.2955 |
| farla_full_r4_supervised | 0.3335 | 0.6215 | 0.2460 |
| farla_full_direct_supervised | 0.3460 | 0.6623 | 0.3120 |
| farla_full_r8_pseudo (no labels in training) | 0.2710 | 0.4698 | 0.2115 |

Planned paired contrasts (macro paired item difference, 95% paired
class-stratified percentile bootstrap, 10,000 replicates, seed 2026091605):

| Contrast | Difference | 95% CI |
|---|---:|---:|
| farla_full_r8 vs no_correction | +0.2345 | [0.2025, 0.2663] |
| farla_full_r8 vs lowrank_ce_r8 | +0.0795 | [0.0473, 0.1115] |
| farla_full_r8 vs lowrank_tail_r8 | -0.0145 | [-0.0405, 0.0113] |
| lowrank_ce_r8 vs shared_translation | +0.1535 | [0.1270, 0.1803] |
| farla_full_r8 vs farla_full_r4 | +0.0170 | [-0.0065, 0.0400] |

Frozen decision in the report: `CONTINUE_PAPER_CANDIDATE` (clean gate,
positive macro gain, and the two decision-required intervals above zero).

## 2. Scientific boundary (read before the numbers)

Sign-off covers the numbers and their custody, not any broader claim. The
boundary below is part of what you are asked to confirm.

1. This is a supervised, 20-shot-per-class, class-specific positive control.
   Every adapter candidate except `farla_full_r8_pseudo` is trained with the
   ground-truth labels of the training role. It is not a zero-shot method and
   must not be described as one. Only `no_correction` is zero-shot.
2. The tail-only control (`lowrank_tail_r8_supervised`, 0.3650) outperforms
   full FARLA (`farla_full_r8_supervised`, 0.3505) on the primary metric, and
   the paired interval for FARLA minus tail-only includes zero. The full FARLA
   objective (hubness plus stable-error-repair terms) is therefore not shown to
   be necessary. The frozen decision rule never tested this contrast; it
   compared FARLA only with `no_correction` and the noisy-CE control.
3. Macro clean accuracy rises for the rank-8 low-rank variants (noisy-CE
   +11.6 points, tail-only +9.9, full FARLA +7.7 over `no_correction`), so part
   of the certified-accuracy gain reflects supervision improving the
   classifier, not only certificate-specific repair. The clean gate in the
   protocol bounds clean-accuracy *drops*; it does not separate these effects.
4. This is not a rescue of EXP-017. EXP-017 is an immutable negative result
   for small shared/global translations; FARLA uses fresh reserved items, a
   different (supervised) protocol, and different candidates. Nothing here
   changes EXP-017's status or the negative-results paper's claims.
5. Scale: four cells, 1,000 CIFAR-100 and 100 EuroSAT confirmation items per
   model. One EuroSAT item is one percentage point. This is an exploratory
   pilot, not a broad replication, and no result may be used to alter the
   frozen rule or to select among candidates after the fact.

## 3. Package contents

| File | Purpose |
|---|---|
| `START_HERE.md` | this guide and the response template |
| `manifest.sha256.json` | SHA-256 and byte size of all 61 files (848,017,691 bytes) under the result directory; recorded CONFIG_SHA256; config-snapshot digest check |
| `build_manifest.py` | regenerates the manifest (stdlib only) |
| `reconstruct_farla_results.py` | independent reconstruction; imports no FARLA code (numpy, scipy, torch for `.pt` deserialisation, stdlib) |
| `reconstruction_report.json` | full machine-readable output of the reconstruction as run by Person A (232 checks) |
| `reconstruction_report.md` | the same, rendered: PASS/FAIL tables, contrasts, data roles, failures, items not verified |
| `build_package_zip.py` | rebuilds the zip and the external manifest |
| `tests/test_farla_result_custody_v1.py` (repository `tests/`) | pytest checks: manifest coverage, hash recompute, config digest, independence of the script, headline aggregates within 1e-9, saved report consistency, zip integrity |

The frozen result directory itself is not in the package (it is 848 MB and
gitignored). You need a copy of `FARLA/downloaded_results/EXP-20260917-020-FARLA-FULL/`
(the two RAR parts next to it are the transfer archives). Step 1 proves that
your copy is the one reviewed here.

## 4. What to run

All commands are run from the repository root on Windows. Use
`.venv-gate-n2/Scripts/python` for anything that opens `.pt` files (torch
2.11 is installed there); `.venv/Scripts/python` suffices for the manifest.
Do not modify anything under `FARLA/downloaded_results/`, `FARLA/results/`,
`results/`, `artifacts/`, or `configs/`.

Step 1: prove the result tree is the reviewed one.

    .venv/Scripts/python FARLA/artifacts/phase4/FARLA_FULL_RESULT_CUSTODY_V1/build_manifest.py --out %TEMP%\farla_manifest_check.json

Compare every `path`, `bytes`, and `sha256` in your output with
`manifest.sha256.json` (the `generated_utc`, `generator`, and `mtime_utc`
fields legitimately differ). The pytest step does this comparison for you.
Expected anchors:

- `CONFIG_SHA256` content: `562608ac874240bf2350bb29c821e0797505d16884d147e253227f8b659ac885`
  (this is the SHA-256 of the canonical JSON form of the configuration, keys
  sorted and compact separators, which is how FARLA identifies a configuration;
  the file digest of `config_snapshot.json` is different by construction:
  `57b79923fd333c01fb128e7dc88f81f9aefce7e6b3aafff4d373d5c5d5511620`, and it
  is byte-identical to `FARLA/configs/phase4/farla_full_v1.json`).
- `analysis/report.json`: `dd123e618475ebb498583fb569e26040f45cf051b6d8338224d61af37a2777c4`
- `splits/assignments.csv`: `eb58533a736c75b25dee351ca4ba4e654f7c4e629ee8bbca2494524a4762d8ca`
- Phase-3 source ledger `artifacts/day14/phase3_splits_v1/assignments.csv`
  and the EXP-017 snapshot `results/EXP-20260906-017/phase3_split_assignments_snapshot.csv`
  (byte-identical): `b98d298282238c2825a519526c00e76c719911e1492c53361b35ea5aeafb1f38`
- `confirmation/*/sufficient_statistics.pt`: `31201d15...` (B/32 CIFAR-100),
  `8c7a9573...` (B/32 EuroSAT), `3e4b6902...` (L/14 CIFAR-100),
  `a0ea3c52...` (L/14 EuroSAT); full digests are in the manifest.

Note: `config_snapshot.json` carries a later modification time (2026-09-21)
than the rest of the tree (2026-09-19). Its bytes were verified identical to
the repository configuration and its canonical digest equals CONFIG_SHA256;
the timestamp is a copy artifact and is recorded as informational only.

Step 2: run the independent reconstruction (about 70 s including the five
bootstraps; add `--bootstrap none` for a 10 s run without them).

    .venv-gate-n2/Scripts/python FARLA/artifacts/phase4/FARLA_FULL_RESULT_CUSTODY_V1/reconstruct_farla_results.py --out-dir %TEMP%\farla_recon

It prints the PASS/FAIL table, the contrasts, the recomputed decision, and
`OVERALL: PASS|FAIL`; exit status is nonzero on any FAIL. Compare your
`reconstruction_report.md` with the packaged one. Writing to a temporary
directory keeps the packaged report as Person A left it; if you prefer to
overwrite it, that is also fine, but then rebuild the zip afterwards.

Step 3: run the custody tests (the live reconstruction test needs torch).

    .venv-gate-n2/Scripts/python -m pytest -q tests/test_farla_result_custody_v1.py

Expected: 7 passed (the zip test skips only if the zip is absent).

Step 4: confirm independence yourself.

    findstr /N /R "^import ^from" FARLA\artifacts\phase4\FARLA_FULL_RESULT_CUSTODY_V1\reconstruct_farla_results.py

Only numpy, scipy.stats, torch (inside `load_pt`), and the standard library
should appear; no `phase4`, `analysis`, `certification`, or `vlm_pipeline`
import and no `sys.path` manipulation. Do not use FARLA's
`scripts/phase4/analyze_results.py` as the verification: it is the code under
review.

## 5. What to compare, and what Person A found

Person A's run (`reconstruction_report.md`): overall PASS, 0 FAIL, 4
NOT_VERIFIED, 232 checks, maximum absolute deviation 1.1e-16 over every
numeric comparison.

1. Aggregates: all eight candidates match `report.json` on every aggregate
   field (macro clean, smoothed, standard CA, anchored CA, certified-wrong,
   average radius, clean-drop points, worst-cell drop, clean gate) within
   1e-9; the five headline candidates also match the 4-decimal values quoted
   above within rounding.
2. Cells: for every cell and candidate, the selected class, selected-class
   successes, Clopper-Pearson lower bound, radius, abstention flag, and the
   per-radius standard/anchored/certified-wrong flags recompute item by item
   from the stored 128/4096 counts; `metrics.json` and the `cells` rows of
   `report.json` recompute within 1e-9. The integer threshold k >= 3518 agrees
   everywhere with the floating-point radius test, and the closest
   non-abstained radius is 2.4e-5 away from 0.25, so scipy-version differences
   in `beta.ppf` (observed at most 2.2e-16) cannot flip any certificate.
3. Contrasts: all five planned contrasts reproduce the reported point
   differences, per-cell exact McNemar counts and p-values, Holm adjustment,
   and the bootstrap intervals exactly (deviation 0.0), using an independent
   re-implementation seeded with 2026091605 and 10,000 replicates.
4. Decision: the frozen rule recomputes to `CONTINUE_PAPER_CANDIDATE`, with
   the boundary in section 2 unchanged.
5. Data roles: 20/5/10 items per class per dataset; roles disjoint by item ID
   and dataset index; every Phase-4 item is `reserved_not_accessed` in the
   Phase-3 ledger; zero confirmation (and zero train/selection) items carry
   any Phase-3 development, validation, excluded, or sealed-test role; the
   whole Phase-4 allocation re-derives from the Phase-3 reserve with the
   documented rank hash; the train and selection feature caches contain
   exactly the ledger's train and selection items and no confirmation item;
   the confirmation item order, labels, and per-item seeds recompute from the
   ledger and the configuration seeds.
6. Bank provenance: each trained checkpoint's prototype bank equals the bank
   used in confirmation exactly, the frozen zero-shot bank equals the cache
   prototypes, and the checkpoint variant specifications and summaries equal
   the configuration and `metrics.json`.

What could not be verified from the outputs (also listed in the report):

- The Monte-Carlo vote counts themselves (they require the GPU encoders,
  checkpoints, and images); only budgets, seeds, and downstream arithmetic.
- The per-candidate raw-clean predictions: they are stored and used as given,
  because the clean features of the confirmation items are not stored.
- Adapter training (80 epochs) is not re-run; only checkpoint-to-confirmation
  linkage and metadata.
- `screening_report.json` (diagnostic only; `candidate_selection_performed`
  is false) is hashed, not reconstructed.
- The `prototype_sha256` field's formula is undocumented; the direct tensor
  equality check stands in for it (the four NOT_VERIFIED entries).
- Server-side execution facts (`preflight.json`, `run.log`, `supervisord.log`)
  are hashed, not re-executed.

## 6. Response template

Copy into `research/independent_review/farla_full_result_custody_v1_signoff_<date>.md`
(or return it by any channel) and fill every field. Leave no field blank.

    FARLA_FULL_RESULT_CUSTODY_V1 independent review

    Reviewer: 
    Date: 
    Zip SHA-256 reviewed (from FARLA_FULL_RESULT_CUSTODY_V1.external_manifest.json): 
    Repository commit / working-tree state: 
    Interpreters used (paths, Python / numpy / scipy / torch versions): 

    Step 1 manifest recompute: PASS / FAIL  (files compared: 61; mismatches: )
    Step 2 reconstruction: PASS / FAIL  (overall status: ; FAIL count: ; max abs deviation: )
    Step 3 pytest: passed / failed / skipped counts: 
    Step 4 independence check: PASS / FAIL  (notes: )

    Headline values reproduced by me (macro std CA@0.25 / clean acc):
      no_correction:                     /
      shared_translation_ce_supervised:  /
      lowrank_ce_r8_supervised:          /
      lowrank_tail_r8_supervised:        /
      farla_full_r8_supervised:          /
    Contrasts reproduced by me (point, 95% CI):
      farla_full_r8 vs no_correction:
      farla_full_r8 vs lowrank_tail_r8:
    Data-role verdict (disjoint roles; no EXP-017 dev/val/excluded/sealed item among confirmation items): CONFIRMED / NOT CONFIRMED
    Items I could not verify beyond those listed in section 5: 

    Boundary acknowledgement (initial each):
      [ ] supervised 20-shot class-specific positive control, not zero-shot
      [ ] tail-only control outperforms full FARLA; full objective not shown necessary
      [ ] clean accuracy rises 8-12 points for rank-8 low-rank variants; part of the gain reflects supervision
      [ ] not an EXP-017 rescue; EXP-017 remains immutable
      [ ] exploratory four-cell pilot; no post hoc candidate selection

    Verdict (exactly one):
      [ ] PASS - the numbers, custody, and data roles are confirmed; the boundary statements above are the only approved framing.
      [ ] PASS WITH CORRECTION - confirmed except for the corrections listed below, which must be applied before any use:
          Corrections required: 
      [ ] FAIL - not confirmed. Blocking findings: 

    Approved wording for the claims matrix (row C10) and paper, if any: 

    Signature: 

Until a completed template with a PASS or PASS WITH CORRECTION verdict exists,
the FARLA numbers stay at their current claims-matrix status (independent
result-custody sign-off pending) and must not enter the paper.
