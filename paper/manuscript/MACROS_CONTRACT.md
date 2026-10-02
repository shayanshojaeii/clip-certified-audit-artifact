# Generated-number contract for the SaTML 2027 manuscript

Every number in `paper/satml2027/` prose and tables must come from
`paper/satml2027/generated/macros.tex` or a generated table file. The
generator is `scripts/generate_satml2027_paper_assets.py`; it reads only
immutable artifacts and records their SHA-256 in
`paper/satml2027/generated/sources.json`. Nobody types a result value by hand.

Percent values are emitted as percentage points with two decimals unless noted
(for example `11.47`), differences carry an explicit sign (`+2.97`, `-0.17`),
and counts are plain integers with thousands separators (`3,300`).

## Macro names (LaTeX commands, no arguments)

EXP-017 scope (from `results/EXP-20260906-017/summary.json` and the soundness audit):
- `\ExpCells` (12), `\ExpPositions` (3,300), `\ExpRows` (59,400), `\ExpSelVotes`, `\ExpConfVotes`, `\ExpBanks` (216), `\ExpCandidates` (18), `\ExpDistinctClassifiers` (15), `\ExpSelDraws` (128), `\ExpConfDraws` (4,096), `\ExpAlpha` (0.001), `\ExpSigma` (0.25), `\ExpCifarItemsPerCell` (500), `\ExpEurosatItemsPerCell` (50)
- `\ExpElapsedHours` (`elapsed_seconds`/3600 from `summary.json`, one decimal), `\SelectedComparator` (`selected_primary_comparator_id` from `summary.json`, underscores escaped)
- `\ExpRadiusPerCoordinate`: 0.25 / sqrt(3 x 224 x 224) with the RGB channel count and `crop_size` read from `configs/clean_pipeline_v1.json` and `configs/clean_pipeline_vit_l14_v1.json`; two significant digits emitted as `m\times 10^{e}` for math mode; derivation recorded in `sources.json`

Equal-cell macro aggregates at pixel-space radius 0.25 (soundness audit `candidate_aggregates_equal_cell_weight`):
- no correction: `\NoCorrStdCA`, `\NoCorrAnchCA`, `\NoCorrClean`, `\NoCorrSmoothed`, `\NoCorrAbstain`, `\NoCorrAgreement`
- saved selection (clean_boundary_active step 0.0025): `\SavedStdCA`, `\SavedAnchCA`, `\SavedClean`
- exact-rule selection (clean_boundary_active step 0.04): `\ExactStdCA`, `\ExactAnchCA`, `\ExactClean`
- GR-CLIP-style two-sided coefficient 1: `\GRStdCA`, `\GRAnchCA`, `\GRClean`, `\GRAgreement`
- global mean-centering coefficient 1: `\GMCStdCA`, `\GMCAnchCA`, `\GMCClean`, `\GMCAgreement`
- standard minus anchored gap over the 18 candidates: `\StdAnchGapMin`, `\StdAnchGapMax` (points, two decimals, unsigned because standard CA is at least anchored CA by construction; checked against the audit's `standard_minus_anchored`)

Gate-N2 simultaneous analysis (`artifacts/negative_paper/gate_n2_simultaneous_analysis_v1.json`), radius 0.25, standard metric unless suffixed `Anch`:
- `\BandHalfWidth` (common critical value in points, 1.98), `\BandReplicates` (100,000), `\BandContrasts` (136), `\BandStrata` (330), `\BandExclZero` (8), `\BandExclZeroGR` (6), `\BandExclZeroOther` (2; the generator refuses unless no bank other than GR-CLIP excludes zero at radius 0.25, every GR-CLIP exclusion is positive, and every other exclusion is a negative standard-metric global-mean-centering contrast at radius 0)
- `\GRStdDelta`, `\GRStdLower`, `\GRStdUpper`, `\GRAnchDelta`, `\GRAnchLower`, `\GRAnchUpper`
- `\SavedStdDelta`, `\SavedStdLower`, `\SavedStdUpper`, `\ExactStdDelta`, `\ExactStdLower`, `\ExactStdUpper`
- `\GMCStdDelta`, `\GMCStdLower`, `\GMCStdUpper`, `\GMCStdDeltaRZero`, `\GMCStdLowerRZero`, `\GMCStdUpperRZero`, `\GMCAnchDeltaRZero`, `\GMCAnchLowerRZero`, `\GMCAnchUpperRZero` (anchored contrast at radius 0)
- `\ChowersMaxAbsDelta` (maximum absolute contrast over the three Chowers coefficients and all radii/metrics; expected 0.00)
- finite-stratum sensitivity (descriptive, not registered): `\BandHalfWidthSens` (= BandHalfWidth x sqrt(5/4)), `\GRStdLowerSens`, `\GRStdUpperSens`

Weighting sensitivity (descriptive; computed from `per_candidate_cells` with each cell's `example_count`):
- `\GRStdDeltaPooled`, `\GMCStdDeltaPooled`, `\SavedStdDeltaPooled`, `\ExactStdDeltaPooled`, `\KishDesignEffect`, `\EffectiveN`
- dataset split (descriptive): `\GRStdDeltaCifar`, `\GRStdDeltaEurosat` (equal-cell mean over the six cells of each dataset of the GR-CLIP minus no-correction standard-CA change at 0.25, signed points), `\GRStdCifarCellsNegative` (CIFAR-100 cells whose change is negative, by integer counts)

Clean gate adjudication (from `results/EXP-20260906-017/selection.json` and `artifacts/gate3/phase3_exp017_negative_diagnostics_v1.json`):
- `\GRWorstCellDrop` (4.00 points), `\GRWorstCell` (cell id, text), `\GRWorstCellImages` (2), `\GRCellsImproved` (9 of 12 -> emit `\GRCellsImproved` = 9), `\FloatGateArtifact` (`2.0000000000000018`), `\FloatGateRejectedCandidates` (count), `\ForcedComparisonFailures` (3)

Failure anatomy for no correction at radius 0.25, pooled over 3,300 positions (compute from the immutable per-example sufficient statistics used by Gate N2; document the source path and hash):
- `\PartAbstain`, `\PartWrongClass`, `\PartCorrectBelowRadius`, `\PartCertifiedCorrect`, `\PartAnchorMismatch`, `\PartAnchoredCorrect`, `\CertifiedWrongRate` (percent of positions with nonabstaining wrong class and radius >= 0.25), `\PartCertifiedWrong` (the same as an integer count), `\CertifiedCorrectAnchorMismatch` (= PartCertifiedCorrect - PartAnchoredCorrect)
- `\CWTopClassShareLo`, `\CWTopClassShareHi`: per cell, the share (percent, one decimal) of no-correction certified-wrong positions at 0.25 whose selected class is the single most frequent selected class among them; minimum and maximum over the cells with at least one such position; `\CWCellsWithNone` counts the excluded cells with zero certified-wrong positions
- changed standard-CA outcomes versus no correction (positions of 3,300 whose indicator nonabstaining and correct and radius >= 0.25 differs, from the same per-example statistics): `\ChangedPositionsSmallSteps` (maximum over the six proposal candidates with step <= 0.01), `\ChangedPositionsExactStep` (clean_boundary_active step 0.04), `\ChangedPositionsGR` (GR-CLIP-style two-sided coefficient 1), `\ChangedPositionsProposalsMax` (maximum over the ten text-only proposal candidates, both objectives and all five steps)

No-correction per-cell ranges (diagnostics JSON `noise_collapse_no_correction_ranges`, one decimal):
- `\RangeCleanLo`, `\RangeCleanHi`, `\RangeSmoothedLo`, `\RangeSmoothedHi`, `\RangeAgreementLo`, `\RangeAgreementHi`, `\RangeZeroRadiusLo`, `\RangeZeroRadiusHi`, `\RangeTopClassLo`, `\RangeTopClassHi`

Candidate movement (macro-average Frobenius displacement of the prototype bank over the 12 cells, from `results/EXP-20260906-017/cells/*/cell_metrics.json` `prototype_bank_frobenius_displacement`):
- `\DispSaved`, `\DispExact`, `\DispGMCOne`, `\DispGR`

Finite-sample censoring (soundness audit `finite_sample_plugin_only_examples`):
- `\CensoredNoCorrRQuarter`, `\CensoredSavedRQuarter`, `\CensoredNoCorrRZero`

FARLA pilot (`FARLA/downloaded_results/EXP-20260917-020-FARLA-FULL/analysis/report.json`, macro standard CA at 0.25 and macro clean accuracy):
- `\FarlaIdentityStdCA`, `\FarlaSharedStdCA`, `\FarlaLowrankCEStdCA`, `\FarlaTailStdCA`, `\FarlaFullStdCA`, `\FarlaIdentityClean`, `\FarlaLowrankCEClean`, `\FarlaTailClean`, `\FarlaFullClean`
- `\FarlaVsNoCorrDelta`, `\FarlaVsNoCorrLower`, `\FarlaVsNoCorrUpper`, `\FarlaVsTailDelta`, `\FarlaVsTailLower`, `\FarlaVsTailUpper`, `\FarlaCifarConfirmItems` (1000), `\FarlaEurosatConfirmItems` (100), `\FarlaShots` (20)
- certified-wrong rates (`macro_certified_wrong_primary`, percent): `\FarlaIdentityCW`, `\FarlaFullCW`, `\FarlaTailCW`
- pseudo-label variant (`farla_full_r8_pseudo`): `\FarlaPseudoStdCA`, `\FarlaPseudoClean`, `\FarlaPseudoCleanDrop` (`macro_clean_drop_percentage_points`, two decimals, positive means clean accuracy fell), and from `paired_comparisons["farla_full_r8_pseudo"]["bootstrap"]`: `\FarlaPseudoVsNoCorrDelta`, `\FarlaPseudoVsNoCorrLower`, `\FarlaPseudoVsNoCorrUpper` (signed points)

Registered extension (from `configs/satml2027/*.json`):
- `\ExtCellsA` (14), `\ExtPrimaryCellsA` (12), `\ExtCellsB` (18), `\ExtCellsN` (4), `\ExtBanks` (22), `\ExtItemsCifar` (500), `\ExtItemsEurosat` (150), `\ExtUniqueItems` (1,150), `\ExtRegHashA`, `\ExtRegHashB`, `\ExtRegHashN` (first 12 hex characters)

Registered extension results (unblinded under D-127 at the owner's instruction; the registered primary analysis outputs under `results/satml2027_local/analysis/` are primary-only; read together with the receipt `results/satml2027_downloads/EXP019_N3C_20260925T172040Z/server_results_root/scientific_run_attempt4_receipt.txt`, the three `results/satml2027_local/merged/*/merge_report.json` files, and `scientific_results/items/item_manifest.json`):
- custody: `\ExtAReceiptStatus` (receipt `status`), `\ExtResultFiles` (`result_file_count`), `\ExtResultManifest` (first 12 hex of `result_manifest_sha256`), `\ExtCellsMerged` (cells over the three complete merge reports), `\ExtControlLabelsPerClass` (`control_train.per_class` from the item manifest, whose item hashes must equal the registrations' `control_train_items_sha256`)
- per experiment (EXP-019A over its 12 primary cells, EXP-019B over its 18 cells; deltas are equal-cell macro changes of standard CA at r = sigma versus no correction in points from `cells.csv`; bands come from `contrasts.json` and satisfy lower/upper = point -/+ `critical_max_absolute_deviation`). Stems: Critical (band half width, points); GRDelta/GRLower/GRUpper/GRCleanDelta (gr_clip_style_two_sided__coefficient_1; the clean delta is the equal-cell clean-accuracy change in points); LowrankDelta/LowrankLower/LowrankUpper/LowrankCleanDelta (control__lowrank_tangent_r8); NoisyMeanDelta/NoisyMeanCleanDelta (control__noisy_class_mean); SharedTangentDelta/SharedTangentLower/SharedTangentUpper (control__learned_shared_translation_tangent); MaxGridDelta/MaxGridId (largest delta among the 17 grid banks; id with escaped underscores)/MinGridDelta; GridExclZero/ControlsExclZero (contrasts whose band excludes zero); Xthree ("confirmed" when the learned shared-translation tangent control's interval contains zero); Xfour ("confirmed" when at least one per-class control, noisy class mean or rank-8 tangent adapter, has a lower limit above zero); XfourPositiveControls (how many of the two per-class controls have a lower limit above zero, in words; the A2.3 flags reported beside the existential verdict); Xfive ("supported" when the identity bank's top-class share and predicted-class Gini both strictly increase with sigma within every model x dataset, bridge cells included, otherwise "not uniformly supported": outcome-blind addendum A1, descriptive, no inferential statement); the summaries use "confirmed / not confirmed" for X3 and X4 and "supported / not uniformly supported" for X5
- EXP-019A names: `\ExtACritical`, `\ExtAGRDelta`, `\ExtAGRLower`, `\ExtAGRUpper`, `\ExtAGRCleanDelta`, `\ExtALowrankDelta`, `\ExtALowrankLower`, `\ExtALowrankUpper`, `\ExtALowrankCleanDelta`, `\ExtANoisyMeanDelta`, `\ExtANoisyMeanCleanDelta`, `\ExtASharedTangentDelta`, `\ExtASharedTangentLower`, `\ExtASharedTangentUpper`, `\ExtAMaxGridDelta`, `\ExtAMaxGridId`, `\ExtAMinGridDelta`, `\ExtAGridExclZero`, `\ExtAControlsExclZero`, `\ExtAXthree`, `\ExtAXfour`, `\ExtAXfive`
- EXP-019B names: `\ExtBCritical`, `\ExtBGRDelta`, `\ExtBGRLower`, `\ExtBGRUpper`, `\ExtBGRCleanDelta`, `\ExtBLowrankDelta`, `\ExtBLowrankLower`, `\ExtBLowrankUpper`, `\ExtBLowrankCleanDelta`, `\ExtBNoisyMeanDelta`, `\ExtBNoisyMeanCleanDelta`, `\ExtBSharedTangentDelta`, `\ExtBSharedTangentLower`, `\ExtBSharedTangentUpper`, `\ExtBMaxGridDelta`, `\ExtBMaxGridId`, `\ExtBMinGridDelta`, `\ExtBGridExclZero`, `\ExtBControlsExclZero`, `\ExtBXthree`, `\ExtBXfour`, `\ExtBXfive`
- bridge cells (CIFAR-10 at sigma 0.25, EXP-019A): `\ExtBridgeGRDeltaBthirtytwo`, `\ExtBridgeGRDeltaLfourteen` (per-cell GR-CLIP minus no-correction standard CA at r = sigma, points)
- identity regime ranges (min and max over the identity-bank cells at that sigma across the available registered experiments, percent): `\ExtSmoothedTwelveCifarLo`, `\ExtSmoothedTwelveCifarHi`, `\ExtStdCATwelveCifarLo`, `\ExtStdCATwelveCifarHi`, `\ExtSmoothedTwelveEurosatLo`, `\ExtSmoothedTwelveEurosatHi` (sigma 0.12; CIFAR-10 and CIFAR-100 cells together), and `\ExtSmoothedFiftyCifarLo`, `\ExtSmoothedFiftyCifarHi`, `\ExtStdCAFiftyCifarLo`, `\ExtStdCAFiftyCifarHi`, `\ExtSmoothedFiftyEurosatLo`, `\ExtSmoothedFiftyEurosatHi` (sigma 0.5)

N3C fresh-noise diagnostic (`results/satml2027_local/analysis/N3C-20260920-V3/n3c_predictions.json`; rule thresholds are parsed from the registration's prediction `rule` strings):
- `\NthreeCPone` (P1 exceptions: flips outside the eligible set summed over applicable candidates and cells), `\NthreeCPfourA` ("max MACE (pass/fail)" over the two CIFAR-100 cells; components `\NthreeCPfourAMax`, `\NthreeCPfourAVerdict`), `\NthreeCPfive` ("k of 4 (pass/fail)" cells with Spearman rho > 0; components `\NthreeCPfivePositive`, `\NthreeCPfiveVerdict`)
- `\NthreeCPseven` (items already at k_min under the identity bank, pooled over the four cells; the per-candidate remainder is in `table_n3c.tex`), `\NthreeCPsevenPositions`, `\NthreeCPsevenCandidates` (candidates with an applicable P7 partition), `\NthreeCPsevenUnattainableLo`, `\NthreeCPsevenUnattainableHi`, `\NthreeCPsevenUndeterminedLo`, `\NthreeCPsevenUndeterminedHi` (ranges over those candidates)
- `\NthreeCPnine` (useful / harmful draw-level winner movements for the two-sided bank and the rank-8 control, pooled; components `\NthreeCPnineTwoSidedUseful`, `\NthreeCPnineTwoSidedHarmful`, `\NthreeCPnineLowrankUseful`, `\NthreeCPnineLowrankHarmful`)

## Generated table files

- `generated/table_family_r025.tex`: all 18 candidates: family, step/coefficient, clean accuracy, standard CA, anchored CA, standard delta with simultaneous band, historical float eligibility, exact-count eligibility. Sorted by family then step.
- `generated/table_partition.tex`: failure partition for no correction and GR-CLIP at radius 0.25 under both metrics.
- `generated/table_grclip_cells.tex` (full width, `table*`): 12 cells: model, dataset, fold, no-correction clean, GR-CLIP clean, drop in points, no-correction standard CA, GR-CLIP standard CA, no-correction anchored CA, GR-CLIP anchored CA. Per-cell standard and anchored CA come from the audit's `per_candidate_cells` (`standard_certified_accuracy["0.25"]`, `anchored_certified_accuracy["0.25"]`); the `certified_accuracy` field of `cell_metrics.json` is the historical anchored metric and must never be labelled standard CA. The column means equal `\NoCorrStdCA`, `\GRStdCA`, `\NoCorrAnchCA`, `\GRAnchCA`.
- `generated/table_farla.tex`: all eight FARLA candidates labelled (identity, shared translation, low-rank CE, low-rank tail, full r=4, full r=8, full direct, and the pseudo-label row marked "pseudo-labels (no ground truth)"): standard CA, anchored CA, clean accuracy, certified-wrong rate, clean-gate status, with the two planned contrasts and their intervals.
- `generated/table_extension.tex`: registered extension cells, items, sigmas, banks, controls, primary estimand, status.
- `generated/table_ext_regime.tex` (`tab:ext-regime`, table*): identity bank per registered cell of EXP-019A (14) and EXP-019B (18): model, dataset, sigma, n, clean accuracy, smoothed accuracy, standard CA at r = sigma, certified-wrong rate at r = sigma, abstention, top-class share (`concentration.csv`); the two CIFAR-10 sigma = 0.25 bridge cells carry a dagger.
- `generated/table_ext_contrasts.tex` (`tab:ext-contrasts`, table*): per experiment, every registered contrast versus no correction (17 grid banks then 4 controls): equal-cell delta of standard CA at r = sigma (points), the registered simultaneous band, the equal-cell clean-accuracy delta (points), and whether the band excludes zero. Names: grid banks as in the family table except `gr_clip_style_two_sided__coefficient_1` = "Two-sided centring c=1"; controls "Learned shared translation (tangent)", "Learned shared translation (pure)", "Noisy class mean (supervised)", "Rank-8 tangent adapter (supervised)". The caption states the controls' label budget per class.
- `generated/table_n3c.tex` (`tab:n3c`, table*): per candidate pooled over the four N3C cells: P1 exceptions, P7 unattainable / undetermined / realized, P9 useful and harmful counts.

- Added after the reviews of V11 (D-159, D-160): `\ExtXfourFlags` (one phrase with the two A2.3 flags beside the existential X4 verdict); `\FUQSixMaxAbsBound` (largest absolute bound, two decimals, of the unadjusted 95% intervals of the registered primary contrasts in both pooled families of EXP-021B) and `\FUQSixMaxUpper` (their largest upper bound, signed); both descriptive, from `analysis/exp021/021b/analysis_ext.json` (questions.Q6.pooled__sigma*.<contrast>.ci95_unadjusted).

Table labels are fixed by the manuscript: `tab:family`, `tab:partition`, `tab:grclip-cells`, `tab:farla`, `tab:extension`, `tab:ext-regime`, `tab:ext-contrasts`, `tab:n3c`. Every table is centred, set in scriptsize, uses booktabs rules, and places its caption before the tabular; the family, per-cell and extension tables are full-width (table*); percentages are emitted without the percent sign.

Every generated file starts with a comment listing its source artifacts and SHA-256 values.
