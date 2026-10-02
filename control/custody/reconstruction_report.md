# Independent reconstruction report: EXP-20260917-020-FARLA-FULL

- Overall status: **PASS** (0 FAIL, 4 NOT_VERIFIED, 232 checks)
- Generated (UTC): 2026-09-24T20:55:35.775706+00:00; elapsed 66 s
- Environment: Python 3.12.14, numpy 2.4.0, scipy 1.18.1, torch 2.11.0+cu130
- Script: `FARLA/artifacts/phase4/FARLA_FULL_RESULT_CUSTODY_V1/reconstruct_farla_results.py` SHA-256 `22dd73d2dd7271f591e8bfa9c3fd2612ce66777c8d15e6b2d62cb1f43f674c73`; imports FARLA code: False
- Results directory: `FARLA/downloaded_results/EXP-20260917-020-FARLA-FULL`; CONFIG_SHA256 `562608ac874240bf2350bb29c821e0797505d16884d147e253227f8b659ac885`
- Comparison tolerance: 1e-09; maximum absolute deviation over all numeric checks: 1.110e-16
- Protocol: sigma 0.25, alpha 0.001, 128 selection / 4096 confirmation draws, primary radius 0.25 (k_min = 3518), bootstrap 10000 x seed 2026091605 (mode: planned)

## Aggregate comparison with analysis/report.json (macro = equal-cell mean over 4 cells)

| Candidate | Std CA@0.25 recon | report | Clean recon | report | Anchored CA recon | report | max abs dev (all fields) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| no_correction | 0.116000 | 0.116000 | 0.608750 | 0.608750 | 0.084250 | 0.084250 | 5.551e-17 | PASS |
| shared_translation_ce_supervised | 0.117500 | 0.117500 | 0.612750 | 0.612750 | 0.086750 | 0.086750 | 0.000e+00 | PASS |
| lowrank_ce_r8_supervised | 0.271000 | 0.271000 | 0.725000 | 0.725000 | 0.233500 | 0.233500 | 2.776e-17 | PASS |
| lowrank_tail_r8_supervised | 0.365000 | 0.365000 | 0.708000 | 0.708000 | 0.287500 | 0.287500 | 5.551e-17 | PASS |
| farla_full_r4_supervised | 0.333500 | 0.333500 | 0.621500 | 0.621500 | 0.246000 | 0.246000 | 0.000e+00 | PASS |
| farla_full_r8_supervised | 0.350500 | 0.350500 | 0.685750 | 0.685750 | 0.295500 | 0.295500 | 5.551e-17 | PASS |
| farla_full_direct_supervised | 0.346000 | 0.346000 | 0.662250 | 0.662250 | 0.312000 | 0.312000 | 0.000e+00 | PASS |
| farla_full_r8_pseudo | 0.271000 | 0.271000 | 0.469750 | 0.469750 | 0.211500 | 0.211500 | 0.000e+00 | PASS |

## Other macro quantities (recomputed)

| Candidate | Smoothed acc | Abstention | Certified wrong@0.25 | Avg selected radius | Clean change vs no_correction (points) | Worst-cell clean drop (points) | Clean gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| no_correction | 0.2237 | 0.1810 | 0.3417 | 0.2775 | -0.000 | +0.00 | True |
| shared_translation_ce_supervised | 0.2162 | 0.1618 | 0.3342 | 0.2849 | +0.400 | +1.00 | True |
| lowrank_ce_r8_supervised | 0.4460 | 0.2060 | 0.1358 | 0.2440 | +11.625 | -5.10 | True |
| lowrank_tail_r8_supervised | 0.5433 | 0.2055 | 0.0905 | 0.2797 | +9.925 | -4.30 | True |
| farla_full_r4_supervised | 0.5160 | 0.2472 | 0.1177 | 0.2753 | +1.275 | +1.00 | True |
| farla_full_r8_supervised | 0.5455 | 0.1943 | 0.1042 | 0.2835 | +7.700 | -3.10 | True |
| farla_full_direct_supervised | 0.4898 | 0.1732 | 0.1143 | 0.2821 | +5.350 | +2.60 | False |
| farla_full_r8_pseudo | 0.4205 | 0.2165 | 0.1775 | 0.2654 | -13.900 | +20.00 | False |

## Per-cell reconstruction (standard CA@0.25 / clean accuracy / abstentions / items with k >= k_min)

| Candidate | openai-clip-vit-b32-quickgelu__cifar100 | openai-clip-vit-b32-quickgelu__eurosat | openai-clip-vit-l14-quickgelu__cifar100 | openai-clip-vit-l14-quickgelu__eurosat |
|---|---:|---:|---:|---:|
| no_correction | 0.084 / 0.595 / 274/1000 / 342 | 0.040 / 0.460 / 16/100 / 33 | 0.250 / 0.710 / 250/1000 / 369 | 0.090 / 0.670 / 4/100 / 79 |
| shared_translation_ce_supervised | 0.089 / 0.609 / 267/1000 / 333 | 0.040 / 0.450 / 10/100 / 31 | 0.251 / 0.722 / 250/1000 / 374 | 0.090 / 0.670 / 3/100 / 79 |
| lowrank_ce_r8_supervised | 0.128 / 0.649 / 400/1000 / 226 | 0.330 / 0.660 / 7/100 / 51 | 0.296 / 0.761 / 244/1000 / 401 | 0.330 / 0.830 / 11/100 / 49 |
| lowrank_tail_r8_supervised | 0.181 / 0.638 / 419/1000 / 243 | 0.470 / 0.670 / 9/100 / 59 | 0.319 / 0.754 / 253/1000 / 399 | 0.490 / 0.770 / 6/100 / 59 |
| farla_full_r4_supervised | 0.142 / 0.585 / 457/1000 / 203 | 0.390 / 0.500 / 16/100 / 59 | 0.302 / 0.711 / 272/1000 / 362 | 0.500 / 0.690 / 10/100 / 65 |
| farla_full_r8_supervised | 0.177 / 0.626 / 397/1000 / 242 | 0.440 / 0.630 / 8/100 / 59 | 0.305 / 0.747 / 250/1000 / 387 | 0.480 / 0.740 / 5/100 / 60 |
| farla_full_direct_supervised | 0.176 / 0.569 / 271/1000 / 310 | 0.460 / 0.660 / 9/100 / 55 | 0.268 / 0.720 / 222/1000 / 431 | 0.480 / 0.700 / 11/100 / 55 |
| farla_full_r8_pseudo | 0.149 / 0.523 / 425/1000 / 218 | 0.280 / 0.260 / 9/100 / 57 | 0.275 / 0.576 / 241/1000 / 396 | 0.380 / 0.520 / 11/100 / 61 |

## Planned contrasts (paired items, macro over cells)

| Contrast | Proposed - comparator (recon) | report | Bootstrap 95% CI (recon) | report CI | Bootstrap status | Requested in task |
|---|---:|---:|---:|---:|:---:|:---:|
| farla_vs_no_correction (farla_full_r8_supervised vs no_correction) | +0.23450 | +0.23450 | [0.20250, 0.26625] | [0.20250, 0.26625] | PASS | True |
| farla_vs_lowrank_ce (farla_full_r8_supervised vs lowrank_ce_r8_supervised) | +0.07950 | +0.07950 | [0.04725, 0.11150] | [0.04725, 0.11150] | PASS | False |
| farla_vs_lowrank_tail (farla_full_r8_supervised vs lowrank_tail_r8_supervised) | -0.01450 | -0.01450 | [-0.04051, 0.01125] | [-0.04051, 0.01125] | PASS | True |
| lowrank_ce_vs_shared_translation (lowrank_ce_r8_supervised vs shared_translation_ce_supervised) | +0.15350 | +0.15350 | [0.12700, 0.18025] | [0.12700, 0.18025] | PASS | False |
| farla_r8_vs_r4 (farla_full_r8_supervised vs farla_full_r4_supervised) | +0.01700 | +0.01700 | [-0.00650, 0.04001] | [-0.00650, 0.04001] | PASS | False |

### Per-cell exact McNemar (proposed-only / comparator-only / p / Holm p)

- farla_vs_no_correction: vit-b32-quickgelu/cifar100: 126/33/5.27e-14/2.11e-13; vit-b32-quickgelu/eurosat: 43/3/4.62e-10/1.39e-09; vit-l14-quickgelu/cifar100: 125/70/9.99e-05/9.99e-05; vit-l14-quickgelu/eurosat: 42/3/8.65e-10/1.73e-09
- farla_vs_lowrank_ce: vit-b32-quickgelu/cifar100: 84/35/8.18e-06/3.27e-05; vit-b32-quickgelu/eurosat: 26/15/1.17e-01/2.35e-01; vit-l14-quickgelu/cifar100: 84/75/5.26e-01/5.26e-01; vit-l14-quickgelu/eurosat: 20/5/4.08e-03/1.22e-02
- farla_vs_lowrank_tail: vit-b32-quickgelu/cifar100: 26/30/6.89e-01/1.00e+00; vit-b32-quickgelu/eurosat: 15/18/7.28e-01/1.00e+00; vit-l14-quickgelu/cifar100: 48/62/2.15e-01/8.60e-01; vit-l14-quickgelu/eurosat: 9/10/1.00e+00/1.00e+00
- lowrank_ce_vs_shared_translation: vit-b32-quickgelu/cifar100: 55/16/3.75e-06/1.13e-05; vit-b32-quickgelu/eurosat: 30/1/2.98e-08/1.19e-07; vit-l14-quickgelu/cifar100: 72/27/6.90e-06/1.38e-05; vit-l14-quickgelu/eurosat: 27/3/8.43e-06/1.38e-05
- farla_r8_vs_r4: vit-b32-quickgelu/cifar100: 54/19/5.06e-05/2.02e-04; vit-b32-quickgelu/eurosat: 8/3/2.27e-01/6.80e-01; vit-l14-quickgelu/cifar100: 47/44/8.34e-01/1.00e+00; vit-l14-quickgelu/eurosat: 5/7/7.74e-01/1.00e+00

## Frozen decision rule (recomputed)

- Recomputed decision: **CONTINUE_PAPER_CANDIDATE**; report decision: **CONTINUE_PAPER_CANDIDATE** (interval source: recomputed bootstrap)
- clean_constraint = True
- positive_macro_gain = True
- paired_interval_above_zero:farla_vs_no_correction = True
- paired_interval_above_zero:farla_vs_lowrank_ce = True
- The frozen rule compares the primary candidate with no_correction and the matched rank-8 noisy-CE adapter only. It does not test the tail-only control; farla_vs_lowrank_tail is reported as a planned contrast outside the decision.

## Scientific boundary facts (numbers only)

- All adapter candidates except farla_full_r8_pseudo are trained with ground-truth labels of the 20-per-class train role (supervised few-shot); no candidate is zero-shot except no_correction.
- Tail-only rank-8 macro standard CA@0.25 = 0.3650 vs full FARLA rank-8 = 0.3505; FARLA - tail = -0.0145, 95% CI [-0.0405, 0.0112]. tail-only rank-8 exceeds full FARLA rank-8 on the primary metric; the full objective is therefore not shown to be necessary.
- Clean-accuracy change vs no_correction (macro percentage points): shared_translation_ce_supervised +0.40, lowrank_ce_r8_supervised +11.62, lowrank_tail_r8_supervised +9.93, farla_full_r4_supervised +1.27, farla_full_r8_supervised +7.70, farla_full_direct_supervised +5.35, farla_full_r8_pseudo -13.90
- Certified-wrong-class rate at 0.25 (macro): no_correction 0.3417, shared_translation_ce_supervised 0.3342, lowrank_ce_r8_supervised 0.1358, lowrank_tail_r8_supervised 0.0905, farla_full_r4_supervised 0.1177, farla_full_r8_supervised 0.1042, farla_full_direct_supervised 0.1143, farla_full_r8_pseudo 0.1775
- EuroSAT confirmation cells hold 100 items each; one item is one percentage point there.

## Data roles

- cifar100.train: 2000 items
- cifar100.selection: 500 items
- cifar100.confirmation: 1000 items
- eurosat.train: 200 items
- eurosat.selection: 50 items
- eurosat.confirmation: 100 items
- train: 20 items per class; labels used for supervised adapter training (noisy-CE, clean-CE, tail, hubness, displacement, stable-error terms); pseudo variant uses clean zero-shot predictions as targets
- selection: 5 items per class; diagnostic screening only (screening_report.json, candidate_selection_performed=false); every predeclared candidate continued
- confirmation: 10 items per class; fresh 128-draw class selection and 4096-draw probability estimation per item; the only role entering report.json
- Phase-4 allocation re-derived from the Phase-3 reserve matches the ledger: True
- Phase-3 role of every Phase-4 item (count by dataset.role -> phase3 role):
  - cifar100.train: reserved_not_accessed=2000, calibration_development=0, calibration_validation=0, excluded_explicit=0, excluded_prior_access=0, final_test_sealed=0
  - cifar100.selection: reserved_not_accessed=500, calibration_development=0, calibration_validation=0, excluded_explicit=0, excluded_prior_access=0, final_test_sealed=0
  - cifar100.confirmation: reserved_not_accessed=1000, calibration_development=0, calibration_validation=0, excluded_explicit=0, excluded_prior_access=0, final_test_sealed=0
  - eurosat.train: reserved_not_accessed=200, calibration_development=0, calibration_validation=0, excluded_explicit=0, excluded_prior_access=0, final_test_sealed=0
  - eurosat.selection: reserved_not_accessed=50, calibration_development=0, calibration_validation=0, excluded_explicit=0, excluded_prior_access=0, final_test_sealed=0
  - eurosat.confirmation: reserved_not_accessed=100, calibration_development=0, calibration_validation=0, excluded_explicit=0, excluded_prior_access=0, final_test_sealed=0

## Failures

- none

## Not verified by this reconstruction

- Monte-Carlo vote counts (selection_counts, confirmation_counts) cannot be regenerated here: that requires the GPU encoders, the frozen checkpoints and the source images. They are taken as stored; only their budgets (128/4096 per row), seeds and downstream arithmetic are verified.
- raw_predictions (per-candidate raw-clean predictions) are stored and used as given. The clean image features of the confirmation items are not stored anywhere in the result tree (only train/selection caches exist), so the clean argmax cannot be recomputed without re-encoding.
- Adapter training (80 epochs, losses, initialisation) is not re-run. Only the linkage checkpoint bank -> confirmation bank, the variant specification and the summary metadata are verified.
- screening_report.json is a diagnostic over the selection role (candidate_selection_performed=false); it is hashed in the manifest but not reconstructed.
- The bootstrap is an independent re-implementation of the resampling scheme documented in the frozen analysis source (FARLA/analysis/paired_bootstrap.py, SHA-256 9014b90062e3b2982916dc3fb5ebc041547d4957ab67dc75de26c915429852ed in the presampling approval). Exact numeric agreement depends on consuming the RNG stream in the same stratum order; the deviation is reported rather than assumed.
- Server-side execution facts (preflight.json, run.log, supervisord.log) are custody-hashed but cannot be re-executed or independently attested from this machine.
- cache.openai-clip-vit-b32-quickgelu__cifar100.prototype_sha256_recomputes: prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check
- cache.openai-clip-vit-b32-quickgelu__eurosat.prototype_sha256_recomputes: prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check
- cache.openai-clip-vit-l14-quickgelu__cifar100.prototype_sha256_recomputes: prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check
- cache.openai-clip-vit-l14-quickgelu__eurosat.prototype_sha256_recomputes: prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check

## All checks

| Check | Status | Detail |
|---|:---:|---|
| `config.snapshot_canonical_hash_matches_CONFIG_SHA256` | PASS | canonical JSON SHA-256 of config_snapshot.json = 562608ac874240bf2350bb29c821e0797505d16884d147e253227f8b659ac885; CONFIG_SHA256 = 562608ac874240bf2350bb29c821e0797505d16884d147e253227f8b659ac885 |
| `config.repository_config_byte_identical_to_snapshot` | PASS | FARLA/configs/phase4/farla_full_v1.json compared byte-for-byte with config_snapshot.json |
| `config.experiment_id` | PASS | experiment_id = EXP-20260917-020-FARLA-FULL |
| `config.protocol_constants_as_expected` | PASS | sigma, alpha, draws, radii, bootstrap seed/replicates, split seed, role sizes, clean-gate limits |
| `certification.minimum_successes_at_primary_radius` | PASS | smallest k with CP_lower(k, 4096, 0.001) >= Phi(0.25/0.25) is 3518 |
| `splits.assignments_sha256_matches_manifest` | PASS | splits/assignments.csv SHA-256 = eb58533a736c75b25dee351ca4ba4e654f7c4e629ee8bbca2494524a4762d8ca |
| `splits.manifest_config_sha256` | PASS | splits/manifest.json config_sha256 equals CONFIG_SHA256 |
| `splits.manifest_final_test_not_accessed` | PASS | splits/manifest.json declares phase3_final_test_accessed=false |
| `splits.phase4_item_ids_unique` | PASS | 3850 rows, 0 duplicate (dataset, item_id) keys |
| `splits.role_counts_and_per_class_balance` | PASS | every (dataset, role) has class_count x per_class items and every class is balanced |
| `splits.roles_disjoint_by_item_id_and_dataset_index` | PASS | an item appears in exactly one Phase-4 role (by item_id and by dataset_index) |
| `splits.all_rows_marked_reserved_not_accessed` | PASS | phase3_role column is reserved_not_accessed on every Phase-4 row |
| `splits.rank_hash_recomputes` | PASS | phase4_rank_sha256 == sha256('20260916:<dataset_id>:<item_id>') on every row |
| `phase3.ledger_sha256_matches_declared_source:artifacts/day14/phase3_splits_v1/assignments.csv` | PASS | SHA-256 b98d298282238c2825a519526c00e76c719911e1492c53361b35ea5aeafb1f38 vs declared b98d298282238c2825a519526c00e76c719911e1492c53361b35ea5aeafb1f38 (declared source: artifacts/day14/phase3_splits_v1/assignments.csv) |
| `phase3.ledger_sha256_matches_declared_source:results/EXP-20260906-017/phase3_split_assignments_snapshot.csv` | PASS | SHA-256 b98d298282238c2825a519526c00e76c719911e1492c53361b35ea5aeafb1f38 vs declared b98d298282238c2825a519526c00e76c719911e1492c53361b35ea5aeafb1f38 (declared source: artifacts/day14/phase3_splits_v1/assignments.csv) |
| `phase3.exp017_snapshot_identical_to_day14_assignments` | PASS | the EXP-017 assignment snapshot and the day-14 Phase-3 split file are byte-identical |
| `phase3.every_phase4_item_found_in_ledger` | PASS | 0 Phase-4 items missing from artifacts/day14/phase3_splits_v1/assignments.csv |
| `phase3.labels_and_dataset_indices_agree` | PASS | label mismatches=0, dataset_index mismatches=0 |
| `phase3.no_confirmation_item_in_exp017_development_validation_excluded_or_sealed_roles` | PASS | 0 confirmation items carry a non-reserved Phase-3 role |
| `phase3.no_phase4_item_in_any_non_reserved_phase3_role` | PASS | 0 Phase-4 items (any role) carry a non-reserved Phase-3 role |
| `splits.phase4_allocation_rederives_from_phase3_reserve` | PASS | ranking reserved items per class by (rank hash, item_id) and slicing 20/5/10 reproduces splits/assignments.csv exactly |
| `report.config_sha256` | PASS | analysis/report.json config_sha256 equals CONFIG_SHA256 |
| `report.primary_radius` | PASS | report primary_radius = 0.25 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.COMPLETE_marker` | PASS | COMPLETE marker present |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.provenance_hashes` | PASS | metrics.json and sufficient_statistics.pt carry CONFIG_SHA256, experiment, model and dataset IDs |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.final_test_not_accessed` | PASS | both files declare phase3_final_test_accessed=false |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.candidate_ids_match_protocol` | PASS | candidate order = ['no_correction', 'shared_translation_ce_supervised', 'lowrank_ce_r8_supervised', 'lowrank_tail_r8_supervised', 'farla_full_r4_supervised', 'farla_full_r8_supervised', 'farla_full_direct_supervised', 'farla_full_r8_pseudo'] |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.item_ids_equal_confirmation_role_in_ledger_order` | PASS | 1000 confirmation items; ledger confirmation rows = 1000 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.ground_truth_equals_ledger_labels` | PASS | ground_truth tensor equals the label column of the confirmation rows |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.per_item_seeds_recompute_from_config_seeds` | PASS | selection/confirmation seeds equal derive_seed(config seed, model, dataset, item, purpose) and never collide |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.vote_budgets` | PASS | every (candidate, item) row sums to 128 selection and 4096 confirmation draws |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.raw_predictions_shape` | PASS | raw_predictions shape = [8, 1000] (per-candidate raw-clean predictions are stored) |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.no_correction.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.no_correction.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 2.776e-17 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.no_correction.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 2.776e-17 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.shared_translation_ce_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.shared_translation_ce_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.shared_translation_ce_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.lowrank_ce_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.lowrank_ce_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.lowrank_ce_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.lowrank_tail_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.lowrank_tail_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.lowrank_tail_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r4_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r4_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r4_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_direct_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_direct_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_direct_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r8_pseudo.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r8_pseudo.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r8_pseudo.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `bank.openai-clip-vit-b32-quickgelu__cifar100.shared_translation_ce_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__cifar100.lowrank_ce_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__cifar100.lowrank_tail_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r4_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__cifar100.farla_full_direct_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__cifar100.farla_full_r8_pseudo.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `cache.openai-clip-vit-b32-quickgelu__cifar100.train.items_equal_ledger_role_and_disjoint_from_confirmation` | PASS | train cache holds exactly the ledger's train items (2000), labels and indices agree, 0 overlap with confirmation items |
| `cache.openai-clip-vit-b32-quickgelu__cifar100.train.frozen_prototypes_equal_no_correction_bank` | PASS | cache prototypes equal candidate_banks[no_correction] exactly |
| `cache.openai-clip-vit-b32-quickgelu__cifar100.prototype_sha256_recomputes` | NOT_VERIFIED | prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check |
| `cache.openai-clip-vit-b32-quickgelu__cifar100.selection.items_equal_ledger_role_and_disjoint_from_confirmation` | PASS | selection cache holds exactly the ledger's selection items (500), labels and indices agree, 0 overlap with confirmation items |
| `cache.openai-clip-vit-b32-quickgelu__cifar100.selection.frozen_prototypes_equal_no_correction_bank` | PASS | cache prototypes equal candidate_banks[no_correction] exactly |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.COMPLETE_marker` | PASS | COMPLETE marker present |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.provenance_hashes` | PASS | metrics.json and sufficient_statistics.pt carry CONFIG_SHA256, experiment, model and dataset IDs |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.final_test_not_accessed` | PASS | both files declare phase3_final_test_accessed=false |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.candidate_ids_match_protocol` | PASS | candidate order = ['no_correction', 'shared_translation_ce_supervised', 'lowrank_ce_r8_supervised', 'lowrank_tail_r8_supervised', 'farla_full_r4_supervised', 'farla_full_r8_supervised', 'farla_full_direct_supervised', 'farla_full_r8_pseudo'] |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.item_ids_equal_confirmation_role_in_ledger_order` | PASS | 100 confirmation items; ledger confirmation rows = 100 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.ground_truth_equals_ledger_labels` | PASS | ground_truth tensor equals the label column of the confirmation rows |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.per_item_seeds_recompute_from_config_seeds` | PASS | selection/confirmation seeds equal derive_seed(config seed, model, dataset, item, purpose) and never collide |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.vote_budgets` | PASS | every (candidate, item) row sums to 128 selection and 4096 confirmation draws |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.raw_predictions_shape` | PASS | raw_predictions shape = [8, 100] (per-candidate raw-clean predictions are stored) |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.no_correction.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.no_correction.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.no_correction.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.shared_translation_ce_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.shared_translation_ce_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.shared_translation_ce_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.lowrank_ce_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.lowrank_ce_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.lowrank_ce_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.lowrank_tail_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.lowrank_tail_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.lowrank_tail_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r4_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r4_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r4_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 5.551e-17 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 5.551e-17 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_direct_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_direct_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 5.551e-17 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_direct_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 5.551e-17 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r8_pseudo.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r8_pseudo.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r8_pseudo.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `bank.openai-clip-vit-b32-quickgelu__eurosat.shared_translation_ce_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__eurosat.lowrank_ce_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__eurosat.lowrank_tail_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r4_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__eurosat.farla_full_direct_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-b32-quickgelu__eurosat.farla_full_r8_pseudo.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `cache.openai-clip-vit-b32-quickgelu__eurosat.train.items_equal_ledger_role_and_disjoint_from_confirmation` | PASS | train cache holds exactly the ledger's train items (200), labels and indices agree, 0 overlap with confirmation items |
| `cache.openai-clip-vit-b32-quickgelu__eurosat.train.frozen_prototypes_equal_no_correction_bank` | PASS | cache prototypes equal candidate_banks[no_correction] exactly |
| `cache.openai-clip-vit-b32-quickgelu__eurosat.prototype_sha256_recomputes` | NOT_VERIFIED | prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check |
| `cache.openai-clip-vit-b32-quickgelu__eurosat.selection.items_equal_ledger_role_and_disjoint_from_confirmation` | PASS | selection cache holds exactly the ledger's selection items (50), labels and indices agree, 0 overlap with confirmation items |
| `cache.openai-clip-vit-b32-quickgelu__eurosat.selection.frozen_prototypes_equal_no_correction_bank` | PASS | cache prototypes equal candidate_banks[no_correction] exactly |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.COMPLETE_marker` | PASS | COMPLETE marker present |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.provenance_hashes` | PASS | metrics.json and sufficient_statistics.pt carry CONFIG_SHA256, experiment, model and dataset IDs |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.final_test_not_accessed` | PASS | both files declare phase3_final_test_accessed=false |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.candidate_ids_match_protocol` | PASS | candidate order = ['no_correction', 'shared_translation_ce_supervised', 'lowrank_ce_r8_supervised', 'lowrank_tail_r8_supervised', 'farla_full_r4_supervised', 'farla_full_r8_supervised', 'farla_full_direct_supervised', 'farla_full_r8_pseudo'] |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.item_ids_equal_confirmation_role_in_ledger_order` | PASS | 1000 confirmation items; ledger confirmation rows = 1000 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.ground_truth_equals_ledger_labels` | PASS | ground_truth tensor equals the label column of the confirmation rows |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.per_item_seeds_recompute_from_config_seeds` | PASS | selection/confirmation seeds equal derive_seed(config seed, model, dataset, item, purpose) and never collide |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.vote_budgets` | PASS | every (candidate, item) row sums to 128 selection and 4096 confirmation draws |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.raw_predictions_shape` | PASS | raw_predictions shape = [8, 1000] (per-candidate raw-clean predictions are stored) |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.no_correction.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.no_correction.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 2.776e-17 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.no_correction.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 2.776e-17 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.shared_translation_ce_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.shared_translation_ce_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.shared_translation_ce_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.lowrank_ce_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.lowrank_ce_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.lowrank_ce_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.lowrank_tail_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.lowrank_tail_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 2.776e-17 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.lowrank_tail_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 2.776e-17 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r4_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r4_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r4_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_direct_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_direct_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 5.551e-17 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_direct_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 5.551e-17 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r8_pseudo.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r8_pseudo.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 2.776e-17 |
| `cell.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r8_pseudo.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 2.776e-17 |
| `bank.openai-clip-vit-l14-quickgelu__cifar100.shared_translation_ce_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__cifar100.lowrank_ce_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__cifar100.lowrank_tail_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r4_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__cifar100.farla_full_direct_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__cifar100.farla_full_r8_pseudo.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `cache.openai-clip-vit-l14-quickgelu__cifar100.train.items_equal_ledger_role_and_disjoint_from_confirmation` | PASS | train cache holds exactly the ledger's train items (2000), labels and indices agree, 0 overlap with confirmation items |
| `cache.openai-clip-vit-l14-quickgelu__cifar100.train.frozen_prototypes_equal_no_correction_bank` | PASS | cache prototypes equal candidate_banks[no_correction] exactly |
| `cache.openai-clip-vit-l14-quickgelu__cifar100.prototype_sha256_recomputes` | NOT_VERIFIED | prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check |
| `cache.openai-clip-vit-l14-quickgelu__cifar100.selection.items_equal_ledger_role_and_disjoint_from_confirmation` | PASS | selection cache holds exactly the ledger's selection items (500), labels and indices agree, 0 overlap with confirmation items |
| `cache.openai-clip-vit-l14-quickgelu__cifar100.selection.frozen_prototypes_equal_no_correction_bank` | PASS | cache prototypes equal candidate_banks[no_correction] exactly |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.COMPLETE_marker` | PASS | COMPLETE marker present |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.provenance_hashes` | PASS | metrics.json and sufficient_statistics.pt carry CONFIG_SHA256, experiment, model and dataset IDs |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.final_test_not_accessed` | PASS | both files declare phase3_final_test_accessed=false |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.candidate_ids_match_protocol` | PASS | candidate order = ['no_correction', 'shared_translation_ce_supervised', 'lowrank_ce_r8_supervised', 'lowrank_tail_r8_supervised', 'farla_full_r4_supervised', 'farla_full_r8_supervised', 'farla_full_direct_supervised', 'farla_full_r8_pseudo'] |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.item_ids_equal_confirmation_role_in_ledger_order` | PASS | 100 confirmation items; ledger confirmation rows = 100 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.ground_truth_equals_ledger_labels` | PASS | ground_truth tensor equals the label column of the confirmation rows |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.per_item_seeds_recompute_from_config_seeds` | PASS | selection/confirmation seeds equal derive_seed(config seed, model, dataset, item, purpose) and never collide |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.vote_budgets` | PASS | every (candidate, item) row sums to 128 selection and 4096 confirmation draws |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.raw_predictions_shape` | PASS | raw_predictions shape = [8, 100] (per-candidate raw-clean predictions are stored) |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.no_correction.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.no_correction.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 1.110e-16 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.no_correction.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 1.110e-16 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.shared_translation_ce_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.shared_translation_ce_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.shared_translation_ce_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.lowrank_ce_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.lowrank_ce_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 1.110e-16 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.lowrank_ce_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 1.110e-16 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.lowrank_tail_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.lowrank_tail_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 5.551e-17 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.lowrank_tail_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 5.551e-17 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r4_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r4_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r4_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r8_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r8_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 5.551e-17 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r8_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 5.551e-17 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_direct_supervised.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_direct_supervised.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_direct_supervised.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r8_pseudo.per_item_outcomes_match_stored` | PASS | selected class, successes, CP lower bound, radius, abstention and per-radius flags recompute item by item |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r8_pseudo.metrics_json_recomputes` | PASS | maximum absolute deviation from metrics.json = 0.000e+00 |
| `cell.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r8_pseudo.report_cell_row_recomputes` | PASS | maximum absolute deviation from report.json cells row = 0.000e+00 |
| `bank.openai-clip-vit-l14-quickgelu__eurosat.shared_translation_ce_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__eurosat.lowrank_ce_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__eurosat.lowrank_tail_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r4_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r8_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__eurosat.farla_full_direct_supervised.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `bank.openai-clip-vit-l14-quickgelu__eurosat.farla_full_r8_pseudo.checkpoint_bank_equals_confirmation_bank` | PASS | trained/<cell>/<variant>.pt candidate_prototypes equal candidate_banks[index] exactly; variant spec equals config; summary equals metrics.json metadata |
| `cache.openai-clip-vit-l14-quickgelu__eurosat.train.items_equal_ledger_role_and_disjoint_from_confirmation` | PASS | train cache holds exactly the ledger's train items (200), labels and indices agree, 0 overlap with confirmation items |
| `cache.openai-clip-vit-l14-quickgelu__eurosat.train.frozen_prototypes_equal_no_correction_bank` | PASS | cache prototypes equal candidate_banks[no_correction] exactly |
| `cache.openai-clip-vit-l14-quickgelu__eurosat.prototype_sha256_recomputes` | NOT_VERIFIED | prototype_sha256 formula is not documented in the outputs; the tensor equality above is the effective check |
| `cache.openai-clip-vit-l14-quickgelu__eurosat.selection.items_equal_ledger_role_and_disjoint_from_confirmation` | PASS | selection cache holds exactly the ledger's selection items (50), labels and indices agree, 0 overlap with confirmation items |
| `cache.openai-clip-vit-l14-quickgelu__eurosat.selection.frozen_prototypes_equal_no_correction_bank` | PASS | cache prototypes equal candidate_banks[no_correction] exactly |
| `aggregate.no_correction.matches_report` | PASS | maximum absolute deviation over all aggregate fields = 5.551e-17 |
| `aggregate.no_correction.matches_task_stated_headline_within_rounding` | PASS | deviation from the 4-decimal values quoted in the custody task = 5.00e-05 |
| `aggregate.shared_translation_ce_supervised.matches_report` | PASS | maximum absolute deviation over all aggregate fields = 0.000e+00 |
| `aggregate.shared_translation_ce_supervised.matches_task_stated_headline_within_rounding` | PASS | deviation from the 4-decimal values quoted in the custody task = 5.00e-05 |
| `aggregate.lowrank_ce_r8_supervised.matches_report` | PASS | maximum absolute deviation over all aggregate fields = 2.776e-17 |
| `aggregate.lowrank_ce_r8_supervised.matches_task_stated_headline_within_rounding` | PASS | deviation from the 4-decimal values quoted in the custody task = 1.11e-16 |
| `aggregate.lowrank_tail_r8_supervised.matches_report` | PASS | maximum absolute deviation over all aggregate fields = 5.551e-17 |
| `aggregate.lowrank_tail_r8_supervised.matches_task_stated_headline_within_rounding` | PASS | deviation from the 4-decimal values quoted in the custody task = 1.11e-16 |
| `aggregate.farla_full_r4_supervised.matches_report` | PASS | maximum absolute deviation over all aggregate fields = 0.000e+00 |
| `aggregate.farla_full_r8_supervised.matches_report` | PASS | maximum absolute deviation over all aggregate fields = 5.551e-17 |
| `aggregate.farla_full_r8_supervised.matches_task_stated_headline_within_rounding` | PASS | deviation from the 4-decimal values quoted in the custody task = 5.00e-05 |
| `aggregate.farla_full_direct_supervised.matches_report` | PASS | maximum absolute deviation over all aggregate fields = 0.000e+00 |
| `aggregate.farla_full_r8_pseudo.matches_report` | PASS | maximum absolute deviation over all aggregate fields = 0.000e+00 |
| `contrast.farla_vs_no_correction.point_difference_matches_report` | PASS | macro paired point difference deviation = 0.000e+00 |
| `contrast.farla_vs_no_correction.mcnemar_and_holm_match_report` | PASS | discordant counts, exact p-values and Holm adjustment deviation = 0.000e+00 |
| `contrast.farla_vs_no_correction.bootstrap_interval_reproduces` | PASS | re-implemented paired class-stratified percentile bootstrap (seed 2026091605, 10000 replicates) deviation = 0.000e+00 |
| `contrast.farla_vs_lowrank_ce.point_difference_matches_report` | PASS | macro paired point difference deviation = 0.000e+00 |
| `contrast.farla_vs_lowrank_ce.mcnemar_and_holm_match_report` | PASS | discordant counts, exact p-values and Holm adjustment deviation = 0.000e+00 |
| `contrast.farla_vs_lowrank_ce.bootstrap_interval_reproduces` | PASS | re-implemented paired class-stratified percentile bootstrap (seed 2026091605, 10000 replicates) deviation = 0.000e+00 |
| `contrast.farla_vs_lowrank_tail.point_difference_matches_report` | PASS | macro paired point difference deviation = 0.000e+00 |
| `contrast.farla_vs_lowrank_tail.mcnemar_and_holm_match_report` | PASS | discordant counts, exact p-values and Holm adjustment deviation = 0.000e+00 |
| `contrast.farla_vs_lowrank_tail.bootstrap_interval_reproduces` | PASS | re-implemented paired class-stratified percentile bootstrap (seed 2026091605, 10000 replicates) deviation = 0.000e+00 |
| `contrast.lowrank_ce_vs_shared_translation.point_difference_matches_report` | PASS | macro paired point difference deviation = 0.000e+00 |
| `contrast.lowrank_ce_vs_shared_translation.mcnemar_and_holm_match_report` | PASS | discordant counts, exact p-values and Holm adjustment deviation = 0.000e+00 |
| `contrast.lowrank_ce_vs_shared_translation.bootstrap_interval_reproduces` | PASS | re-implemented paired class-stratified percentile bootstrap (seed 2026091605, 10000 replicates) deviation = 0.000e+00 |
| `contrast.farla_r8_vs_r4.point_difference_matches_report` | PASS | macro paired point difference deviation = 0.000e+00 |
| `contrast.farla_r8_vs_r4.mcnemar_and_holm_match_report` | PASS | discordant counts, exact p-values and Holm adjustment deviation = 0.000e+00 |
| `contrast.farla_r8_vs_r4.bootstrap_interval_reproduces` | PASS | re-implemented paired class-stratified percentile bootstrap (seed 2026091605, 10000 replicates) deviation = 0.000e+00 |
| `decision.frozen_rule_recomputes` | PASS | recomputed decision CONTINUE_PAPER_CANDIDATE; report decision CONTINUE_PAPER_CANDIDATE |
