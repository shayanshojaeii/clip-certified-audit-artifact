# SaTML 2027 Outcome-Blind Analysis Addendum V1

Date: 2026-09-25  
Status: DRAFT FOR INDEPENDENT APPROVAL; OUTCOME-BLIND; NO EXP-019A/B OR N3C RESULT INSPECTED  
Scope: EXP-20260920-019A, EXP-20260920-019B, N3C-20260920-V3 (the unchanged
approved V4 queue, commit `7c26704eb3899893dd385ae3feeace97b078c41a`)  
Authority: this addendum changes no registration, no sampling, no worker, and
no primary analysis. It fixes secondary analyses, reporting language, the
adjudication of one recorded implementation deviation, the outcome-blindness
attestation, and the power statement before any result exists, so that all of
them can be independently approved and hash-bound before unblinding.

## 0. Binding, files, and frozen constants

### 0.1 What the approval binds

Independent approval is recorded in
`research/SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.approved.json`, which
must contain exactly these fields:

| Field | Required value |
|---|---|
| `status` | `INDEPENDENTLY_APPROVED_FOR_OUTCOME_BLIND_ADDENDUM_V1` |
| `addendum_sha256` | SHA-256 of this file as approved |
| `sensitivity_source_sha256` | SHA-256 of `satml2027/day1/d1_09_sensitivity_analysis.py` |
| `tf32_adjudication_source_sha256` | SHA-256 of `satml2027/day1/d1_10_tf32_preparation_adjudication.py` |
| `primary_analysis_source_sha256` | SHA-256 of `satml2027/day1/d1_07_analyze.py` (`546e34a5eb986b341a636581ea9726828176e09df8b864b60ccc23df6bbc63d2` at the approved commit) |
| `new_sampling_authorized`, `equivalence_claim_authorized`, `confirmatory_claim_authorized`, `causal_claim_authorized`, `final_test_authorized` | all `false` |
| `outcome_blind_at_approval` | `true` |
| `review_date`, `reviewer`, `reviewer_conclusion` | free text |

`d1_09_sensitivity_analysis.py` refuses to run unless this file exists, the
status is exact, the addendum and sensitivity-source hashes match the files on
disk, every optional source hash present matches, every authority flag is
`false`, and `sensitivity_v1.json` does not already exist (the guard mirrors
`analysis/negative_paper_gate_n2.py::verify_execution_approval`). Any edit to
this addendum or to `d1_09` after approval invalidates the approval.

### 0.2 New files introduced by this addendum

- `research/SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.md` (this file)
- `satml2027/day1/d1_09_sensitivity_analysis.py` (items A2, A3, A4)
- `satml2027/day1/d1_10_tf32_preparation_adjudication.py` (item A6)
- `satml2027/tests/test_outcome_blind_addendum_v1.py`

No pre-existing file under `satml2027/`, `configs/`, `results/`, or
`artifacts/` is modified. The registered primary analysis
`d1_07_analyze.py` is imported, never edited.

### 0.3 Frozen constants restated from the registrations (verbatim values)

| Quantity | Value | Source |
|---|---|---|
| EXP-019A registration hash | `9802bf3e5f97ac99aa725a1607e2cbe395886bfd16587c9b7d9db0545c94b1e2` | `configs/satml2027/exp-20260920-019a.json` (file SHA-256 `80f744b12f6a8ebf2d76b10a879373ef13b87c86e4c3960062c97b9c7fef749e`) |
| EXP-019B registration hash | `3fcdbade664a3ded421e724060ddf810fe8fbc47f37b1c6d582515dbe9a96c20` | `configs/satml2027/exp-20260920-019b.json` (file SHA-256 `a112916a9c3c8c04d4995864100603c9b95fcf5bf57d60dfadb5afb54a32c193`) |
| N3C registration hash | `96fd42f268d95b48b1cfbffb9afcd28c9dc4cb5c91ce46f17d4b53d75c0f2622` | `configs/satml2027/n3c_v3.json` (file SHA-256 `78ac71a4bec1a1e104d01ed3c0bd8afe7971f21c83a4244ca939722fceb7e15d`) |
| Item manifest hash (all three registrations) | `2a366db63931f9a903ace946800c0092ada96cc2d33f95f93a1b94048a725c8d` | `item_manifest_sha256` |
| Selection / confirmation draws | 128 / 4,096 | `certification` |
| Per-example alpha | 0.001 | `certification.alpha_per_example` |
| Primary radius rule | r = sigma; `k_min_at_r_equals_sigma` = 3,518 | `certification` |
| Reported radii | 0.0, 0.12, 0.25, 0.5 | `certification.reported_radii` |
| Encoder precision text | "float32 (TF32 disabled); a registered precision-equivalence check is reported" | `certification.encoder_precision` |
| Band | 95% simultaneous max-absolute-deviation, 100,000 replicates, seed 2026091605 | `primary_outcome`; `d1_07_analyze.py` constants |
| Practical-effect region | absolute CA 0.005 (+-0.5 point); "reporting region only; not an equivalence margin or TOST" | `prospective_practical_effect_region` |
| Portable bank-reproduction tolerance | 2e-6 (max absolute tensor error) | `candidates.frozen_grid.portable_reproduction_atol` |
| Tie tolerance (direction construction) | 1e-10 | `candidates.direction_construction.tie_tolerance` |
| Item counts per cell | CIFAR-100 500, CIFAR-10 500, EuroSAT 150 (evaluation); N3C 1,000 / 300 (development) | `cells[*].item_count` |

Registered evaluation item lists (`results/satml2027/items/*__evaluation.csv`,
hash-verified against `evaluation_items_sha256` on 2026-09-25): CIFAR-100
`1a7e4960...` (500 items, 100 classes, exactly 5 per class), CIFAR-10
`bef574f4...` (500 items, 10 classes, exactly 50 per class), EuroSAT
`6940e5fb...` (150 items, 10 classes, exactly 15 per class).

## A1. Registered primary analysis (unchanged)

A1.1 The primary analysis is `satml2027/day1/d1_07_analyze.py` at the approved
commit, run exactly as the README states:
`python satml2027/day1/d1_07_analyze.py --merged results/satml2027/merged/<EXP> --registration configs/satml2027/<registration>.json --out analysis/<019a|019b>`,
after `d1_06_merge_verify.py` has proven that the shards tile every registered
cell exactly once and that every vote row sums to the registered draw budget.
Its outputs are `cells.csv` (one row per cell and bank), `macro.csv` (one row
per bank, equal weight per primary cell), `concentration.csv` (hub diagnostics
of the smoothed predictions), and `contrasts.json`. It does not write per-item
outcome tables; per-item outcomes are recomputed from the merged files by the
same `common.certify.certify_from_counts` call whenever needed (A3).

A1.2 Estimand. For experiment E in {019A, 019B}, primary cell set C_E, cell
size n_c, and bank b different from `no_correction`:

```text
CA_c(b)     = (1/n_c) * sum_i 1[ item i is certified-correct under bank b at r = sigma_c ]
Delta_E(b)  = (1/|C_E|) * sum_{c in C_E} [ CA_c(b) - CA_c(no_correction) ]
```

"Certified-correct at r = sigma_c" is the standard (not raw-clean-anchored)
Cohen outcome of the registered CERTIFY arithmetic: the class selected by the
argmax of the 128 selection draws, the Clopper-Pearson lower bound at
alpha = 0.001 on that class over the 4,096 confirmation draws exceeding 1/2 (no
abstention), certified radius sigma_c * Phi^-1(lower bound) >= sigma_c
(equivalently at least 3,518 confirmation successes), and the selected class
equal to the ground-truth label. Every bank is scored on the same encoded
noisy draws, so each contrast is paired at the draw level. Equal weight per
cell; no item pooling in the primary.

A1.3 Primary cell sets. EXP-019A: |C_E| = 12, namely the two models
(`openai-clip-vit-b32-quickgelu`, `openai-clip-vit-l14-quickgelu`) x three
datasets (cifar100, cifar10, eurosat) x sigma in {0.12, 0.5}. The two CIFAR-10
sigma = 0.25 cells of EXP-019A are the registered secondary bridge and are
excluded from the macro and from the band (the inline rule in `d1_07`:
`dataset_id == "cifar10" and sigma == 0.25 and experiment_id endswith "019A"`).
EXP-019B: |C_E| = 18, namely three models (`openai-clip-rn50-quickgelu`,
`openai-clip-vit-b16-quickgelu`, `openclip-vit-b32-laion2b`) x three datasets x
sigma in {0.25, 0.5}; all 18 are primary.

A1.4 Contrast family. Each experiment has exactly one family of 21 contrasts
versus `no_correction`: 17 frozen-grid candidates plus 4 registered controls.
The identifiers, enumerated from the registration grids
(`common_unit_direction_step_grid` = [0.0025, 0.005, 0.01, 0.02, 0.04],
`gap_coefficient_grid` = [0.25, 0.5, 1.0]) exactly as
`common/banks.py::build_eighteen_candidate_banks` names them, are:

Frozen grid (17):
1. `global_mean_centering__coefficient_0.25`
2. `global_mean_centering__coefficient_0.5`
3. `global_mean_centering__coefficient_1`
4. `gr_clip_style_two_sided__coefficient_1`
5. `chowers_exact_projected_gap__coefficient_0.25`
6. `chowers_exact_projected_gap__coefficient_0.5`
7. `chowers_exact_projected_gap__coefficient_1`
8. `clean_boundary_active__step_0.0025`
9. `clean_boundary_active__step_0.005`
10. `clean_boundary_active__step_0.01`
11. `clean_boundary_active__step_0.02`
12. `clean_boundary_active__step_0.04`
13. `cohen_aligned_noisy_margin__step_0.0025`
14. `cohen_aligned_noisy_margin__step_0.005`
15. `cohen_aligned_noisy_margin__step_0.01`
16. `cohen_aligned_noisy_margin__step_0.02`
17. `cohen_aligned_noisy_margin__step_0.04`

Registered controls (4):
18. `control__learned_shared_translation_tangent`
19. `control__learned_shared_translation_pure`
20. `control__noisy_class_mean`
21. `control__lowrank_tangent_r8`

The reference bank `no_correction` completes the 22-bank set that the worker,
merge and N3C validator enforce. `d1_09` derives this family from the
registration file and refuses any merged file whose bank list differs.

A1.5 Band. The registered band is the shared-item, (dataset, class)-stratified
bootstrap of `d1_07::simultaneous_bootstrap`: seed 2026091605, 100,000
replicates, generator `numpy.random.default_rng(seed)` (NumPy's default
PCG64), each stratum resampled with replacement to its own size, one sampled
multiplicity reused in every primary cell containing that item and for every
contrast, critical value = 0.95-quantile (`method="higher"`) over replicates of
max over the 21 contrasts of |Delta* - Delta_hat|, interval = Delta_hat +-
critical. The critical value is common to all 21 contrasts of the family.
EXP-019A and EXP-019B are two separate families with two separate critical
values; they are never pooled, and no joint statement across them is
inferential (A5).

A1.6 Other registered outputs and their status. The per-cell exact McNemar
tests with Holm adjustment across cells inside each contrast, the flag
`point_inside_prospective_practical_effect_region` (a point-estimate flag),
`macro.csv`, `concentration.csv`, all secondary outcomes listed in the
registrations (standard CA at 0.12/0.25/0.5, certified-wrong rate, abstention,
top-class share, predicted-class Gini, normalized prediction entropy, smoothed
accuracy, clean accuracy), and the two EXP-019A bridge cells are reported as
registered and are descriptive: no inferential claim is attached to any of
them. Prediction X1 is the sign-independent report of all 17 frozen-grid
contrasts against the practical-effect region under A2's language rule.
Prediction X5 (top-class share and predicted-class Gini of `no_correction`
increase with sigma) is evaluated on `concentration.csv` per (model, dataset)
as the sign of the change of each statistic from the lowest to the highest
registered sigma (monotonicity across the three EXP-019A CIFAR-10 sigmas); it
is described as "supported" only if both statistics increase in every pair and
otherwise as "not uniformly supported", with no inferential statement. N3C is
governed solely by its own registration (P1 zero exceptions; P4a mean
absolute calibration error <= 0.05 on both CIFAR-100 cells; P5 rho > 0 in at
least 3 of 4 cells; P2, P6, P7, P8, P9 descriptive/structural) and by
`d1_08_n3c_predictions.py`; it issues no certificate, selects no candidate, and
enters no family of this addendum.

## A2. Definition of "band" for predictions X3 and X4

A2.1 "Band" means the registered 95% simultaneous max-absolute-deviation band
of A1.5, read from `contrasts.json` at
`contrasts[<candidate_id>].simultaneous_confidence_interval` (`lower`,
`upper`, `point_difference`) with the common critical value
`simultaneous_band.critical_max_absolute_deviation`. "Stays inside the band"
means that the contrast's simultaneous interval contains zero.

A2.2 X3 is confirmed if and only if the learned shared-translation control's
interval contains zero, i.e. `lower <= 0 <= upper` for
`control__learned_shared_translation_tangent` (the trained form,
`t'_k = normalize(t_k + v - <v,t_k> t_k)`). The unprojected form
`control__learned_shared_translation_pure` of the same vector is reported
beside it with the same flag as a descriptive companion; the X3 verdict does
not depend on it.

A2.3 X4 is confirmed if and only if at least one per-class control's interval
excludes zero on the positive side, i.e. `lower > 0` for at least one of
`control__noisy_class_mean` and `control__lowrank_tangent_r8`. Both flags are
reported.

A2.4 The +-0.5-point practical-effect region (absolute CA 0.005) is a reporting
region only. The phrase "inside the region" (or any synonym implying that an
effect is bounded by the region) may be used for a contrast only if (i) the
band is narrower than the region, `critical_max_absolute_deviation < 0.005`,
and (ii) that contrast's whole interval lies within [-0.005, +0.005]. If the
band is not narrower than the region, the paper states that the band is wider
than the reporting region and reports the interval endpoints without
region-containment language. The registered point flag
`point_inside_prospective_practical_effect_region` is a statement about the
point estimate only and is never quoted as interval containment. Nothing in
this item is an equivalence test or a TOST conclusion.

A2.5 `d1_09` writes these evaluations to `predictions_registered_band` and
the per-contrast flags `contains_zero`, `excludes_zero_positive`,
`interval_within_practical_region`, `band_narrower_than_region`, and
`inside_region_language_permitted` in `sensitivity_v1.json`. The same
evaluation under the A3 rescaled band is written to
`predictions_rao_wu_band_secondary` and is secondary.

## A3. Finite-stratum sensitivity (secondary; registered band remains primary)

A3.1 Strata. The registered bootstrap resamples, within each (dataset, class)
stratum, n_s shared item identities with replacement from the same n_s
identities. Exact per-stratum sizes, computed from the registered evaluation
lists and identical for EXP-019A and EXP-019B:

| Dataset | Items per cell | Classes | Stratum size n_s | Strata | sqrt(n_s/(n_s-1)) | Cells containing each identity (019A / 019B) |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-100 | 500 | 100 | 5 | 100 | 1.118033989 | 4 / 6 |
| CIFAR-10 | 500 | 10 | 50 | 10 | 1.010152545 | 4 / 6 |
| EuroSAT | 150 | 10 | 15 | 10 | 1.035098339 | 4 / 6 |

Per experiment: 120 strata, 1,150 unique identities, 4,600 (019A) or 6,900
(019B) item positions. The naive within-stratum bootstrap has variance
(n_s - 1)/n_s times the unbiased within-stratum variance estimate: 0.80 for
CIFAR-100, 0.98 for CIFAR-10, 0.9333 for EuroSAT. Because stratum sizes differ
across datasets, the sensitivity is implemented per stratum (Rao-Wu
rescaling), not as one global factor.

A3.2 Statistic. Write the equal-cell macro contrast as a sum over identities:
Delta_hat_c = sum_j a_{j,c}, with a_{j,c} = sum over primary cells k containing
j of d_c(j,k) / (n_k |C_E|), where d_c(j,k) in {-1, 0, 1} is the paired
difference of the certified-correct indicator. For a stratified resample with
multiplicities m_{s,j}:

```text
unscaled:  Delta*_c     = Delta_hat_c + sum_s                      sum_j (m_{s,j} - 1) a_{j,c}
Rao-Wu:    Delta*_RW,c  = Delta_hat_c + sum_s sqrt(n_s/(n_s - 1)) * sum_j (m_{s,j} - 1) a_{j,c}
```

The unscaled form is algebraically the registered replicate statistic. The
Rao-Wu form multiplies each stratum's centered replicate contribution by
sqrt(n_s/(n_s - 1)); its rescaled weights 1 + sqrt(n_s/(n_s-1)) (m_{s,j} - 1)
sum to n_s within every stratum, so cell denominators are unchanged. Critical
values are the 0.95 `higher` quantiles over replicates of max_c |Delta* -
Delta_hat| for each form. Singleton strata (none exist in these registrations)
would receive factor 1 and be counted in `singleton_strata`.

A3.3 Implementation. `d1_09_sensitivity_analysis.py` recomputes the per-item
outcomes from the merged files with the same `certify_from_counts` call and
radii as `d1_07`, rebuilds the 21 differences over the same primary cells,
verifies that its point estimates equal the `point_difference` values of
`contrasts.json` to 1e-12 (otherwise it stops: the merged files are not the
ones the primary analysis consumed), verifies the canonical registration hash
and the band metadata of `contrasts.json` (seed 2026091605, 100,000
replicates, 0.95, method string), and then draws one PCG64DXSM sequence
(`numpy.random.Generator(numpy.random.PCG64DXSM(2026091605))`, 100,000
replicates, strata in (dataset, label) order, identities in item-id order,
multiplicities from `integers(0, n_s, size=(100000, n_s))` as in Gate N2) from
which both the unscaled and the Rao-Wu deviations are accumulated. The seed and
replicate count are read from `d1_07`'s constants at run time, so they cannot
drift from the registered analysis. The registered primary band uses NumPy's
default PCG64 stream; the unscaled PCG64DXSM critical value is therefore an
independent Monte-Carlo realization of the same statistic and is reported
under `monte_carlo_agreement_with_registered_critical` as a diagnostic, never
as a replacement.

A3.4 Reported quantities (`sensitivity_v1.json`, written next to
`contrasts.json`): `registered_band.critical_max_absolute_deviation`
(primary), `sensitivity_bootstrap.critical_unscaled`,
`sensitivity_bootstrap.critical_rao_wu`, their ratio, per-contrast
`rao_wu_interval` = point +- critical_rao_wu, per-contrast bootstrap standard
deviations of both forms, and the simple global inflation bound
`global_inflation_bound`: factor max_s sqrt(n_s/(n_s-1)) = sqrt(5/4) =
1.118033989 (CIFAR-100 strata) applied to the registered critical value and
to the unscaled PCG64DXSM critical value, with per-contrast
`global_inflation_bound_interval`. The bound is a bound in the Gaussian limit
(the rescaled covariance is dominated by factor^2 times the unscaled
covariance, and Anderson's inequality then orders the max-abs quantiles); the
exact finite-replicate Rao-Wu computation is the sensitivity itself.

A3.5 Interpretation rule. The registered band decides X3 and X4 and all
interval language (A2). The Rao-Wu band and the global bound are reported in
the supplement as sensitivity; if a registered-band verdict changes under the
Rao-Wu band, the paper states both verdicts, keeps the registered verdict as
primary, and says that the conclusion is sensitive to the finite-stratum
correction. No verdict is chosen after seeing which band is more favorable.

## A4. Weighting sensitivity (descriptive)

For every one of the 21 contrasts, `d1_09` reports beside the equal-cell
primary the item-pooled contrast

```text
Delta_pooled(b) = sum_c n_c * delta_c(b) / sum_c n_c,   delta_c(b) = CA_c(b) - CA_c(no_correction),
```

i.e. the plain mean of d over all primary item positions (4,600 in 019A,
6,900 in 019B). Under item pooling each CIFAR dataset carries 43.5% of the
weight and EuroSAT 13.0%, versus one third each under the equal-cell primary.
The pooled value is labeled descriptive (`item_pooled_point_difference`,
`item_pooled_minus_equal_cell`, and the per-cell `cell_difference` and
`item_count` that reproduce both weightings); it carries no band and supports
no claim.

## A5. Multiplicity statement

Two experiments x 21 contrasts = 42 registered contrasts. Simultaneous
coverage is controlled within each experiment's family of 21 by its own
max-absolute-deviation band; no joint control across the two experiments, the
bridge cells, the secondary outcomes, the additional radii, the per-cell
McNemar tables, or N3C is claimed. Any sentence that compares or combines
EXP-019A and EXP-019B, or that counts confirmations across experiments, is
descriptive. The paper says so in the results section in one sentence.

## A6. TF32 deviation adjudication

A6.1 The deviation. `satml2027/day1/d1_02_build_development.py` line 244 and
`satml2027/day1/d1_03b_train_controls.py` line 173 set only
`torch.backends.cuda.matmul.allow_tf32 = False`; `torch.backends.cudnn.allow_tf32`
keeps PyTorch's default `True`. The registrations state "float32 (TF32
disabled)". On the RTX 4090 the cuDNN convolutions in the image-encoder
forward passes of the preparation (the ViT patch-embedding convolution; every
RN50 convolution) may therefore have used TF32 while computing the development
clean and noisy features (d1_02) and the control-training feature cache
(d1_03b). Consequently affected artifacts: the two proposal directions, the
development image mean, the 17 non-reference frozen-grid banks, the GR-CLIP
calibration image mean, and all four control banks. Expected unaffected: the
text prototypes (`no_correction`; the text encoder has no convolution), item
lists, seeds. The sampling worker `d1_05_run_shard.py` (`_configure_precision`,
lines 58-64) disables both flags, so every selection, confirmation and N3C
sampling pass ran as registered; the preflight `d1_01_preflight.py` also
disables both (lines 140-141), and its synthetic TF32-versus-float32 label
disagreement on a random bank was 0.0 for all five models
(`artifacts/satml2027/server_preflight_v4/results/preflight_A.json`,
operational metadata) - context for the worker path, not evidence about the
preparation artifacts. The approved scripts were executed exactly as
approved; the deviation is between the approved code and the registration
text, and it is disclosed in the paper in either outcome below.

A6.2 Procedure (fixed in advance).

1. On a separate host, from the same approved commit
   (`7c26704eb3899893dd385ae3feeace97b078c41a`; source archive SHA-256
   `8a59ce1ac549c23f5a3262b2125f5572179838dcc0ede21d9ee8be5e97576000`), the
   same three registration files (hashes in 0.3), the same registered item
   CSVs, the same five checkpoints (hash-verified against the server preflight
   record), and the same runtime tuple (torch 2.11.0+cu130, torchvision
   0.26.0+cu130, OpenCLIP 3.3.0, NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0),
   re-run the unchanged preparation launcher
   `satml2027/launch/day1_prepare.sh <ROOT> 0.05` (SHA-256
   `65a4ec81eea6abf06619b7ac01f131e97a2b6e09141b0a3a5ee892e6b6a71f39`) with
   TF32 fully disabled. Mechanism: the environment variable
   `NVIDIA_TF32_OVERRIDE=0` exported for the whole launcher, which forces
   cuBLAS and cuDNN to ignore any programmatic TF32 setting; no approved file
   is edited. The recompute log must record the variable, and an
   override-effectiveness probe run in the same environment must show that a
   float32 matrix product with `allow_tf32 = True` set programmatically
   agrees with a float64 reference at float32 precision (relative error of
   order 1e-7, not 1e-3). Precondition: `d1_05_run_shard.py --rng-selftest`
   on the recompute host must reproduce the server's digests
   `130e128145c57074626f753b96e507da96327424562f9bbb1e471b245f1d2ac4` for
   shape [64, 3, 224, 224] and
   `a2a950b8cc05fc83e84b16338840f16a80f9dd66303ddd366d544e7c1d13c47e` for
   shape [64, 3, 32, 32]; otherwise the recompute is not the same noise
   realization and the comparison is invalid. The recompute is registered in
   `experiments/EXPERIMENT_REGISTRY.csv` as a precision adjudication run
   before launch; it draws no evaluation sample and creates no result
   directory.
2. Optional attribution run, if time permits: the same host, same launcher,
   without the override (server flag state). Comparing it with the TF32-off
   recompute isolates the cuDNN-TF32 effect from host/kernel differences. It
   informs the disclosure text only; it does not enter the decision.
3. The server preparation tree is downloaded only after the scientific queue
   has written its terminal receipt, and is verified against the server
   preparation manifest `day1_prepare_attempt2_receipt.txt.manifest.sha256`
   (252 files; manifest SHA-256
   `ef1303dba4432e8fd838da76c9b4c9032e0c9f7212071477d0f1a8e985c3c102`).
   Before that download, `d1_10 manifest-check` may compare the recompute
   tree's small `.json` sidecars against the manifest hashes: sidecar equality
   already implies equality of every recorded tensor hash.
4. `d1_10_tf32_preparation_adjudication.py compare` produces one JSON
   (`analysis/tf32_adjudication/tf32_preparation_adjudication_v1.json`) with,
   per cell: max-absolute and relative (relative to the reference maximum and
   Frobenius-relative) deviations of the development clean features, the
   development noisy features, both proposal directions, the text prototypes,
   the GR-CLIP image mean, each of the 22 banks, the control parameters
   (`shared_delta`, `lowrank_left`, `lowrank_right`), and the control-training
   feature cache; exact-hash equality per tensor; whether any bank changes
   identity beyond the registered portable tolerance 2e-6; and whether any
   discrete quantity differs: candidate identifiers (the exact 18 + 4 set and
   order), implied deployability flags (both directions present, finite and
   unit-norm, as `d1_02` requires before saving), tie flags (for every
   development item: the protected class, the critical-competitor set under
   the registered tie tolerance 1e-10, and the flag "more than one critical
   competitor, or the protected class tied with its nearest competitor within
   the tolerance"), item identities and labels
   of the development and control-training lists, and every discrete metadata
   field (family, objective, step or coefficient, image transform, role,
   training hyperparameters, registration hash). Development-item argmax
   agreement under each bank is reported as informational only, because the
   worker never scores development items.
5. Decision rule, fixed now: if all 22 banks of all 36 cells (including the
   GR-CLIP calibration mean) agree within 2e-6 and no discrete flag differs,
   EXP-019A/B are reported as executed with a documented immaterial precision
   deviation (`EXECUTED_WITH_IMMATERIAL_PRECISION_DEVIATION`). Otherwise
   EXP-019A/B are downgraded to a descriptive extension
   (`DOWNGRADED_TO_DESCRIPTIVE_EXTENSION`): the registered analysis is still
   run and reported in full and sign-independently, but every X1/X3/X4
   sentence is prefixed as descriptive, no confirmatory language is used, and
   the deviation is quantified in the paper with the maximum bank deviation,
   the number of banks beyond tolerance, and the discrete differences found.
   No rerun of the preparation or of the sampling is performed before the
   SaTML deadline in either case. N3C, whose banks come from the same
   preparation, inherits the same label. Independent review of the JSON and of
   the resulting paper text is required either way before submission.
6. Reviewer note recorded before any adjudication data exists: control banks
   are the output of an 80-epoch AdamW optimization on the affected feature
   cache and may plausibly exceed 2e-6 even if the frozen-grid banks do not;
   the rule above is nevertheless applied as written. Any change to the rule
   must be made in a V2 addendum before the recompute output is inspected.

## A7. Outcome-blindness attestation

A7.1 Inspected so far (2026-09-23 to 2026-09-25, all operational metadata):
process state (guard, launcher and worker PIDs; their liveness); GPU status
(`nvidia-smi` utilization, memory, temperature); worker log progress lines
(items done, encodes per second, ETA) and the absence of traceback or error
signatures; the `done` completion masks and `items_done`/`items_total` of
shard checkpoints and their `.meta.json` sidecars; completed-job counts
(4/36 with the fifth job at 100/500 at the last check); the receipt of the
preparation (status, counts, manifest hash) and the failed attempt-1 launch
receipt; SHA-256 values of the plan, launcher, registrations, preparation
manifest, partial checkpoint files, guard scripts and recovery records; reboot
evidence (`uptime`, `who -b`, `last -x reboot`); the presence or absence of
result directories; and, in the reboot recovery, the check that completed
rows carry nonzero seeds. No `raw_predictions`, `selection_counts`,
`confirmation_counts`, feature store, certificate, radius, vote margin,
metric, `cells.csv`, `macro.csv`, `concentration.csv`, `contrasts.json`, or
any N3C quantity has been opened or computed by anyone on the team.

A7.2 Required before unblinding: each team member who had access to the host
signs the statement below (Markdown, committed under
`research/independent_review/`), and the independent reviewer acknowledges
both before `d1_06`/`d1_07` are run on the retrieved results.

```text
Outcome-blindness statement (SaTML 2027 EXP-019A/B/N3C)
Name:                              Date:
Commit of this addendum:           Addendum SHA-256:
I confirm that between the launch of the approved queue and the date above I
opened, printed, plotted, summarized or computed no scientific outcome of
EXP-20260920-019A, EXP-20260920-019B or N3C-20260920-V3 (no prediction, vote
count, feature, certificate, radius, metric or analysis output), and that the
only files or fields I accessed are those listed in A7.1 of the addendum.
Files or fields I accessed beyond that list (or "none"):
Signature:
```

## A8. Reporting rule

A8.1 Every registered cell that finished is reported regardless of sign,
magnitude, or direction, in `cells.csv` and in the paper's supplement; every
contrast of both families is reported with its point estimate and registered
interval; no candidate, control, cell, radius, or secondary outcome is added,
dropped, reordered, or highlighted after inspection.

A8.2 Unfinished cells are reported as "not run" with the cause (deadline,
interruption). A family's band and macro are computed only if all of its
primary cells finished; an incomplete family is reported cell by cell
(descriptively, with per-cell McNemar tables) with "not run" entries and no
macro or band, and the paper states that the family is incomplete. Bridge
cells and N3C cells that did not finish are likewise "not run". Absence is
never used to select what is reported, and no scope descent is registered
after the fact.

A8.3 Any per-cell-only reporting script needed for an incomplete family is
written and independently reviewed before the corresponding results are
inspected.

## A9. Power statement to be printed in the paper

A9.1 Text for the paper:

> The extension evaluates 500 CIFAR-100, 500 CIFAR-10, and 150 EuroSAT items
> per cell, and the same 1,150 unique items are reused across the model and
> sigma cells of each experiment. Under the planning assumptions of our
> feasibility report (smallest effect of interest +0.02 absolute certified
> accuracy at r = sigma; paired-difference variance d - 0.02^2 with paired
> discordance d; one-sided normal approximation; Bonferroni control at
> 0.05/17; worst-case full within-item dependence across model and sigma
> repeats, so that the effective sample size is the number of unique items),
> 80% power requires 1,281 effective items at d = 0.04, 1,927 at d = 0.06,
> 3,220 at d = 0.10, and 6,453 at d = 0.20. With 1,150 effective items the
> approximate power to detect a 2-point effect is 0.74, 0.51, 0.27, and 0.11
> respectively. The extension is therefore descriptive and not powered to
> detect a 2-point effect; its simultaneous bands are compatibility
> statements, not equivalence tests.

A9.2 Provenance of the numbers. Smallest effect, variance form, Bonferroni
0.05/17, 80% power, the worst-case dependence convention, and the required
effective sizes 1,281 / 1,927 / 3,220 / 6,453 (427 / 643 / 1,074 / 2,151
items per dataset for three datasets; power 0.9995 / 0.9847 / 0.8527 / 0.4728
at 1,200 items per dataset) are taken verbatim from
`research/NEGATIVE_PAPER_GATE_NR0_FEASIBILITY_REPORT.md`, sections 2-3. The
powers at 1,150 effective items (0.7434, 0.5095, 0.2725, 0.1082) are computed
from the same formula, Phi(0.02 * sqrt(1150) / sqrt(d - 0.0004) - z_{1 -
0.05/17}) with z_{1 - 0.05/17} = 2.7543, on 2026-09-25 without any result.
The NR0 family for planning is the 17 candidate contrasts; the registered
21-contrast max-deviation band is not a Bonferroni procedure, and the +-0.005
reporting region of A2 is unrelated to the 0.02 smallest effect of interest.

## Annex B. Verification performed for this addendum (no result touched)

- Stratum sizes, class counts and item-list hashes: recomputed from the
  registered CSVs on 2026-09-25 (CIFAR-100 5 per class; CIFAR-10 50 per
  class; EuroSAT 15 per class; hashes equal the registrations).
- Contrast family: `d1_09::registered_candidate_family` enumerates exactly 21
  contrasts and 22 banks from each of the two EXP-019 registrations; canonical
  registration hashes recompute to the recorded values.
- `d1_07` output format: `cells.csv`, `macro.csv`, `concentration.csv`,
  `contrasts.json` with keys `experiment_id`, `registration_sha256`,
  `reference`, `primary_outcome`, `simultaneous_band{confidence_level,
  replicates, seed, critical_max_absolute_deviation, method}`,
  `prospective_practical_effect_region{lower, upper, interpretation}`,
  `contrasts{<id>: {simultaneous_confidence_interval{point_difference, lower,
  upper}, cell_mcnemar[...], point_inside_prospective_practical_effect_region}}`,
  `final_test_access`. No per-item table is written by `d1_07`.
- Focused tests: `satml2027/tests/test_outcome_blind_addendum_v1.py`
  (synthetic merged and preparation inputs in a temporary directory; guard
  refusal without the approval file; Rao-Wu bootstrap against a naive
  reference; exact factor scaling for uniform strata; TF32 adjudication
  decisions on within-tolerance, beyond-tolerance and discrete-difference
  trees; manifest verification; `.pt` round trip when torch is present).

## Annex C. Locks restated

No new sampling, no EXP-018, no historical final-test access, no broader or
unregistered replication, no outcome-dependent protocol change, and no claim
before independent reconstruction of the complete result chain. This addendum
is subordinate to the registrations, the V4 sign-off, and any later signed
review; where it is silent, the registered protocol governs.
