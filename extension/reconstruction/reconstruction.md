# EXP019_RESULT_REVIEW_20260925_V1: independent reconstruction of EXP-019A/B

Generated 2026-09-25T18:53:34.036280+00:00 from `results/satml2027_downloads/EXP019_N3C_20260925T172040Z` (raw count shards only; merged files and d1_07 outputs were not read except for the comparison section).

Scientific boundary: descriptive registered extension; not powered; no equivalence claim; no causal claim; bridge cells excluded from the primary; every registered cell reported regardless of sign. The Rao-Wu and item-pooled figures are descriptive, unapproved sensitivities.

## EXP-20260920-019A

- registration `9802bf3e5f97ac99aa725a1607e2cbe395886bfd16587c9b7d9db0545c94b1e2` (recomputed; file `80f744b12f6a8ebf2d76b10a879373ef13b87c86e4c3960062c97b9c7fef749e`; match = True)
- alpha 0.001, selection 128, confirmation 4096, k_min at r = sigma recomputed 3518 (registered 3518)
- primary cells (12): openai-clip-vit-b32-quickgelu__cifar100__sigma0.12, openai-clip-vit-b32-quickgelu__cifar100__sigma0.5, openai-clip-vit-b32-quickgelu__cifar10__sigma0.12, openai-clip-vit-b32-quickgelu__cifar10__sigma0.5, openai-clip-vit-b32-quickgelu__eurosat__sigma0.12, openai-clip-vit-b32-quickgelu__eurosat__sigma0.5, openai-clip-vit-l14-quickgelu__cifar100__sigma0.12, openai-clip-vit-l14-quickgelu__cifar100__sigma0.5, openai-clip-vit-l14-quickgelu__cifar10__sigma0.12, openai-clip-vit-l14-quickgelu__cifar10__sigma0.5, openai-clip-vit-l14-quickgelu__eurosat__sigma0.12, openai-clip-vit-l14-quickgelu__eurosat__sigma0.5
- bridge cells excluded from the primary (2): openai-clip-vit-b32-quickgelu__cifar10__sigma0.25, openai-clip-vit-l14-quickgelu__cifar10__sigma0.25
- registered band: numpy.random.default_rng (PCG64), seed 2026091605, 100000 replicates, 120 strata, 1150 unique identities, 4600 positions; critical max-abs deviation = 0.018389

### Primary estimand: equal-cell macro standard CA at r = sigma (reference no_correction = 0.2393)

| bank | macro CA@sigma | delta vs no_correction | lower | upper | contains 0 | excludes 0 (+) | excludes 0 (-) |
|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.2393 | 0.0000 | -0.0184 | 0.0184 | yes | no | no |
| chowers_exact_projected_gap__coefficient_0.5 | 0.2393 | 0.0000 | -0.0184 | 0.0184 | yes | no | no |
| chowers_exact_projected_gap__coefficient_1 | 0.2393 | 0.0000 | -0.0184 | 0.0184 | yes | no | no |
| clean_boundary_active__step_0.0025 | 0.2398 | 0.0006 | -0.0178 | 0.0189 | yes | no | no |
| clean_boundary_active__step_0.005 | 0.2400 | 0.0007 | -0.0177 | 0.0191 | yes | no | no |
| clean_boundary_active__step_0.01 | 0.2400 | 0.0007 | -0.0177 | 0.0191 | yes | no | no |
| clean_boundary_active__step_0.02 | 0.2405 | 0.0012 | -0.0172 | 0.0196 | yes | no | no |
| clean_boundary_active__step_0.04 | 0.2410 | 0.0017 | -0.0167 | 0.0201 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.0025 | 0.2389 | -0.0003 | -0.0187 | 0.0181 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.005 | 0.2397 | 0.0004 | -0.0180 | 0.0188 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.01 | 0.2404 | 0.0011 | -0.0173 | 0.0195 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.02 | 0.2403 | 0.0011 | -0.0173 | 0.0194 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.04 | 0.2424 | 0.0032 | -0.0152 | 0.0216 | yes | no | no |
| control__learned_shared_translation_pure | 0.2067 | -0.0326 | -0.0510 | -0.0142 | no | no | yes |
| control__learned_shared_translation_tangent | 0.2391 | -0.0002 | -0.0186 | 0.0182 | yes | no | no |
| control__lowrank_tangent_r8 | 0.3902 | 0.1509 | 0.1326 | 0.1693 | no | yes | no |
| control__noisy_class_mean | 0.3304 | 0.0912 | 0.0728 | 0.1096 | no | yes | no |
| global_mean_centering__coefficient_0.25 | 0.2379 | -0.0014 | -0.0198 | 0.0170 | yes | no | no |
| global_mean_centering__coefficient_0.5 | 0.2342 | -0.0051 | -0.0235 | 0.0133 | yes | no | no |
| global_mean_centering__coefficient_1 | 0.2137 | -0.0256 | -0.0440 | -0.0072 | no | no | yes |
| gr_clip_style_two_sided__coefficient_1 | 0.2424 | 0.0032 | -0.0152 | 0.0216 | yes | no | no |

### Registered predictions (pre-registered wording)

- **X1** (falsifiable): "report every frozen-grid candidate's macro standard-CA change at r=sigma relative to no correction against the prospective practical-effect region, without an equivalence claim"
  - region: absolute CA 0.005 (reporting region only; not an equivalence margin or TOST); critical = 0.018388888888888844; band narrower than region = False
  - 15/17 point estimates inside the region; 0/17 intervals within the region; 16/17 contain zero; 0/17 exclude zero on the positive side; 1/17 on the negative side
  - outcome: reported: all 17 frozen-grid contrasts listed with point estimate and registered interval; no equivalence claim is made; the band is wider than the +-0.005 reporting region, so no 'inside the region' language is permitted
- **X3** (falsifiable): "the learned shared-translation control stays inside the same band"
  - control__learned_shared_translation_tangent: point -0.0002, interval [-0.0186, +0.0182], contains zero = True
  - companion control__learned_shared_translation_pure: point -0.0326, interval [-0.0510, -0.0142], contains zero = False
  - outcome: confirmed
- **X4** (falsifiable positive control): "the per-class control banks leave the band"
  - control__lowrank_tangent_r8: point +0.1509, interval [+0.1326, +0.1693], lower > 0 = True, upper < 0 = False
  - control__noisy_class_mean: point +0.0912, interval [+0.0728, +0.1096], lower > 0 = True, upper < 0 = False
  - outcome: confirmed
- **X5** (falsifiable): "top-class share and predicted-class Gini of the zero-shot bank increase with sigma"
  - rule: 'supported' only if both statistics increase in every consecutive sigma pair of every (model, dataset); otherwise 'not uniformly supported'; descriptive, no inferential statement
  - openai-clip-vit-b32-quickgelu|cifar100|0.12->0.5: top-class share 0.078 -> 0.564 (up); Gini 0.516 -> 0.935 (up)
  - openai-clip-vit-b32-quickgelu|cifar10|0.12->0.25: top-class share 0.152 -> 0.300 (up); Gini 0.104 -> 0.418 (up)
  - openai-clip-vit-b32-quickgelu|cifar10|0.25->0.5: top-class share 0.300 -> 0.698 (up); Gini 0.418 -> 0.685 (up)
  - openai-clip-vit-b32-quickgelu|eurosat|0.12->0.5: top-class share 0.500 -> 0.973 (up); Gini 0.743 -> 0.895 (up)
  - openai-clip-vit-l14-quickgelu|cifar100|0.12->0.5: top-class share 0.050 -> 0.108 (up); Gini 0.344 -> 0.796 (up)
  - openai-clip-vit-l14-quickgelu|cifar10|0.12->0.25: top-class share 0.116 -> 0.162 (up); Gini 0.056 -> 0.173 (up)
  - openai-clip-vit-l14-quickgelu|cifar10|0.25->0.5: top-class share 0.162 -> 0.264 (up); Gini 0.173 -> 0.414 (up)
  - openai-clip-vit-l14-quickgelu|eurosat|0.12->0.5: top-class share 0.593 -> 1.000 (up); Gini 0.737 -> 0.900 (up)
  - outcome: supported

### Identity bank (no_correction) regime per cell

| cell | sigma | role | clean acc | smoothed acc | abstention | CA@sigma | cert-wrong@sigma | top-class share | Gini | classes predicted |
|---|---|---|---|---|---|---|---|---|---|---|
| openai-clip-vit-b32-quickgelu__cifar100__sigma0.12 | 0.1200 | primary | 0.6100 | 0.4000 | 0.2560 | 0.2600 | 0.1280 | 0.0780 | 0.5163 | 91 |
| openai-clip-vit-b32-quickgelu__cifar100__sigma0.5 | 0.5000 | primary | 0.6100 | 0.0680 | 0.2520 | 0.0300 | 0.3140 | 0.5640 | 0.9354 | 32 |
| openai-clip-vit-b32-quickgelu__cifar10__sigma0.12 | 0.1200 | primary | 0.9000 | 0.7660 | 0.0880 | 0.5840 | 0.0600 | 0.1520 | 0.1040 | 10 |
| openai-clip-vit-b32-quickgelu__cifar10__sigma0.25 | 0.2500 | bridge | 0.9000 | 0.4520 | 0.2020 | 0.2420 | 0.0780 | 0.3000 | 0.4176 | 10 |
| openai-clip-vit-b32-quickgelu__cifar10__sigma0.5 | 0.5000 | primary | 0.9000 | 0.2300 | 0.2140 | 0.0720 | 0.2780 | 0.6980 | 0.6852 | 10 |
| openai-clip-vit-b32-quickgelu__eurosat__sigma0.12 | 0.1200 | primary | 0.4733 | 0.0933 | 0.0600 | 0.0533 | 0.5267 | 0.5000 | 0.7427 | 5 |
| openai-clip-vit-b32-quickgelu__eurosat__sigma0.5 | 0.5000 | primary | 0.4733 | 0.0933 | 0.0067 | 0.0867 | 0.6933 | 0.9733 | 0.8947 | 2 |
| openai-clip-vit-l14-quickgelu__cifar100__sigma0.12 | 0.1200 | primary | 0.7240 | 0.6300 | 0.1620 | 0.4820 | 0.0800 | 0.0500 | 0.3444 | 96 |
| openai-clip-vit-l14-quickgelu__cifar100__sigma0.5 | 0.5000 | primary | 0.7240 | 0.1460 | 0.3620 | 0.0680 | 0.1200 | 0.1080 | 0.7964 | 53 |
| openai-clip-vit-l14-quickgelu__cifar10__sigma0.12 | 0.1200 | primary | 0.9620 | 0.9000 | 0.0440 | 0.8180 | 0.0240 | 0.1160 | 0.0560 | 10 |
| openai-clip-vit-l14-quickgelu__cifar10__sigma0.25 | 0.2500 | bridge | 0.9620 | 0.7160 | 0.1240 | 0.5100 | 0.0480 | 0.1620 | 0.1728 | 10 |
| openai-clip-vit-l14-quickgelu__cifar10__sigma0.5 | 0.5000 | primary | 0.9620 | 0.4120 | 0.2380 | 0.1640 | 0.0920 | 0.2640 | 0.4144 | 10 |
| openai-clip-vit-l14-quickgelu__eurosat__sigma0.12 | 0.1200 | primary | 0.6600 | 0.2667 | 0.0667 | 0.1600 | 0.4200 | 0.5933 | 0.7373 | 6 |
| openai-clip-vit-l14-quickgelu__eurosat__sigma0.5 | 0.5000 | primary | 0.6600 | 0.1000 | 0.0000 | 0.0933 | 0.8933 | 1.0000 | 0.9000 | 1 |

### Sensitivities (descriptive, unapproved sensitivity)

- item-pooled contrast: plain mean of the paired difference over all 4600 primary item positions (CIFAR cells weigh more than EuroSAT); carries no band.
- Rao-Wu finite-stratum band: numpy.random.Generator(numpy.random.PCG64DXSM(seed)), seed 2026091605, 100000 replicates; critical_rao_wu = 0.019146; unscaled PCG64DXSM critical = 0.018444 (Monte-Carlo re-realization of the registered statistic; registered PCG64 critical = 0.018389); ratio = 1.0380; global bound factor sqrt(5/4) = 1.118034

| bank | equal-cell delta | item-pooled delta | pooled - equal-cell | Rao-Wu lower | Rao-Wu upper | RW contains 0 | RW excludes 0 (+) |
|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0000 | 0.0000 | 0.0000 | -0.0191 | 0.0191 | yes | no |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0000 | 0.0000 | 0.0000 | -0.0191 | 0.0191 | yes | no |
| chowers_exact_projected_gap__coefficient_1 | 0.0000 | 0.0000 | 0.0000 | -0.0191 | 0.0191 | yes | no |
| clean_boundary_active__step_0.0025 | 0.0006 | 0.0002 | -0.0003 | -0.0186 | 0.0197 | yes | no |
| clean_boundary_active__step_0.005 | 0.0007 | 0.0004 | -0.0003 | -0.0184 | 0.0199 | yes | no |
| clean_boundary_active__step_0.01 | 0.0007 | 0.0004 | -0.0003 | -0.0184 | 0.0199 | yes | no |
| clean_boundary_active__step_0.02 | 0.0012 | 0.0011 | -0.0001 | -0.0179 | 0.0204 | yes | no |
| clean_boundary_active__step_0.04 | 0.0017 | 0.0017 | 0.0000 | -0.0174 | 0.0209 | yes | no |
| cohen_aligned_noisy_margin__step_0.0025 | -0.0003 | -0.0004 | -0.0001 | -0.0195 | 0.0188 | yes | no |
| cohen_aligned_noisy_margin__step_0.005 | 0.0004 | 0.0000 | -0.0004 | -0.0188 | 0.0195 | yes | no |
| cohen_aligned_noisy_margin__step_0.01 | 0.0011 | 0.0004 | -0.0007 | -0.0180 | 0.0203 | yes | no |
| cohen_aligned_noisy_margin__step_0.02 | 0.0011 | 0.0009 | -0.0002 | -0.0181 | 0.0202 | yes | no |
| cohen_aligned_noisy_margin__step_0.04 | 0.0032 | 0.0026 | -0.0006 | -0.0160 | 0.0223 | yes | no |
| control__learned_shared_translation_pure | -0.0326 | -0.0461 | -0.0135 | -0.0518 | -0.0135 | no | no |
| control__learned_shared_translation_tangent | -0.0002 | 0.0063 | 0.0065 | -0.0194 | 0.0189 | yes | no |
| control__lowrank_tangent_r8 | 0.1509 | 0.0924 | -0.0586 | 0.1318 | 0.1701 | no | yes |
| control__noisy_class_mean | 0.0912 | 0.0428 | -0.0483 | 0.0720 | 0.1103 | no | yes |
| global_mean_centering__coefficient_0.25 | -0.0014 | 0.0002 | 0.0016 | -0.0205 | 0.0178 | yes | no |
| global_mean_centering__coefficient_0.5 | -0.0051 | -0.0057 | -0.0005 | -0.0243 | 0.0140 | yes | no |
| global_mean_centering__coefficient_1 | -0.0256 | -0.0263 | -0.0007 | -0.0448 | -0.0065 | no | no |
| gr_clip_style_two_sided__coefficient_1 | 0.0032 | 0.0026 | -0.0006 | -0.0160 | 0.0223 | yes | no |

### Macro over primary cells (every metric, every bank)

| bank | CA@sigma | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | sel=raw | mean radius | top share | Gini | entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.2393 | 0.3170 | 0.2626 | 0.1963 | 0.0428 | 0.2181 | 0.3024 | 0.1458 | 0.3421 | 0.7216 | 0.3468 | 0.3829 | 0.4247 | 0.5939 | 0.6205 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.2393 | 0.3170 | 0.2626 | 0.1963 | 0.0428 | 0.2181 | 0.3024 | 0.1458 | 0.3421 | 0.7216 | 0.3468 | 0.3829 | 0.4247 | 0.5939 | 0.6205 |
| chowers_exact_projected_gap__coefficient_1 | 0.2393 | 0.3170 | 0.2626 | 0.1963 | 0.0428 | 0.2181 | 0.3024 | 0.1458 | 0.3421 | 0.7216 | 0.3468 | 0.3829 | 0.4247 | 0.5939 | 0.6205 |
| clean_boundary_active__step_0.0025 | 0.2398 | 0.3172 | 0.2633 | 0.1964 | 0.0425 | 0.2181 | 0.3026 | 0.1460 | 0.3418 | 0.7212 | 0.3463 | 0.3832 | 0.4254 | 0.5941 | 0.6199 |
| clean_boundary_active__step_0.005 | 0.2400 | 0.3172 | 0.2639 | 0.1963 | 0.0425 | 0.2182 | 0.3022 | 0.1473 | 0.3427 | 0.7216 | 0.3463 | 0.3834 | 0.4276 | 0.5951 | 0.6185 |
| clean_boundary_active__step_0.01 | 0.2400 | 0.3186 | 0.2641 | 0.1952 | 0.0425 | 0.2171 | 0.3046 | 0.1478 | 0.3422 | 0.7223 | 0.3461 | 0.3837 | 0.4273 | 0.5957 | 0.6183 |
| clean_boundary_active__step_0.02 | 0.2405 | 0.3187 | 0.2658 | 0.1964 | 0.0419 | 0.2178 | 0.3054 | 0.1487 | 0.3420 | 0.7190 | 0.3468 | 0.3842 | 0.4261 | 0.5964 | 0.6169 |
| clean_boundary_active__step_0.04 | 0.2410 | 0.3199 | 0.2660 | 0.1974 | 0.0428 | 0.2171 | 0.3026 | 0.1478 | 0.3435 | 0.7153 | 0.3496 | 0.3843 | 0.4244 | 0.5963 | 0.6178 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.2389 | 0.3170 | 0.2627 | 0.1966 | 0.0425 | 0.2166 | 0.3032 | 0.1478 | 0.3424 | 0.7192 | 0.3460 | 0.3825 | 0.4239 | 0.5934 | 0.6211 |
| cohen_aligned_noisy_margin__step_0.005 | 0.2397 | 0.3175 | 0.2639 | 0.1961 | 0.0423 | 0.2169 | 0.3018 | 0.1483 | 0.3426 | 0.7188 | 0.3458 | 0.3820 | 0.4237 | 0.5939 | 0.6208 |
| cohen_aligned_noisy_margin__step_0.01 | 0.2404 | 0.3187 | 0.2642 | 0.1953 | 0.0425 | 0.2169 | 0.3005 | 0.1478 | 0.3442 | 0.7161 | 0.3463 | 0.3810 | 0.4237 | 0.5935 | 0.6213 |
| cohen_aligned_noisy_margin__step_0.02 | 0.2403 | 0.3210 | 0.2653 | 0.1967 | 0.0421 | 0.2171 | 0.2989 | 0.1488 | 0.3454 | 0.7133 | 0.3495 | 0.3787 | 0.4224 | 0.5923 | 0.6231 |
| cohen_aligned_noisy_margin__step_0.04 | 0.2424 | 0.3214 | 0.2688 | 0.1986 | 0.0422 | 0.2193 | 0.2957 | 0.1549 | 0.3474 | 0.7046 | 0.3500 | 0.3716 | 0.4203 | 0.5911 | 0.6247 |
| control__learned_shared_translation_pure | 0.2067 | 0.2677 | 0.2212 | 0.1706 | 0.0356 | 0.1892 | 0.4044 | 0.0948 | 0.2842 | 0.6070 | 0.3973 | 0.4666 | 0.5368 | 0.7055 | 0.4919 |
| control__learned_shared_translation_tangent | 0.2391 | 0.3387 | 0.2711 | 0.1896 | 0.0398 | 0.2293 | 0.1477 | 0.2147 | 0.3746 | 0.7345 | 0.3797 | 0.2303 | 0.3032 | 0.5191 | 0.7636 |
| control__lowrank_tangent_r8 | 0.3902 | 0.5306 | 0.4516 | 0.3551 | 0.0953 | 0.3717 | 0.0831 | 0.1956 | 0.5674 | 0.8237 | 0.5696 | 0.2772 | 0.1147 | 0.2422 | 0.9556 |
| control__noisy_class_mean | 0.3304 | 0.4683 | 0.3866 | 0.2858 | 0.0833 | 0.1964 | 0.1185 | 0.2101 | 0.5130 | 0.3271 | 0.3024 | 0.2601 | 0.1252 | 0.2471 | 0.9633 |
| global_mean_centering__coefficient_0.25 | 0.2379 | 0.3177 | 0.2640 | 0.1954 | 0.0411 | 0.2185 | 0.3133 | 0.1423 | 0.3382 | 0.7382 | 0.3514 | 0.3966 | 0.4286 | 0.6050 | 0.6108 |
| global_mean_centering__coefficient_0.5 | 0.2342 | 0.3062 | 0.2540 | 0.1881 | 0.0447 | 0.2178 | 0.3021 | 0.1420 | 0.3218 | 0.7069 | 0.3658 | 0.4036 | 0.4054 | 0.6272 | 0.6121 |
| global_mean_centering__coefficient_1 | 0.2137 | 0.2763 | 0.2326 | 0.1738 | 0.0404 | 0.2047 | 0.3359 | 0.1204 | 0.2924 | 0.5984 | 0.4502 | 0.3994 | 0.4381 | 0.6912 | 0.5482 |
| gr_clip_style_two_sided__coefficient_1 | 0.2424 | 0.3262 | 0.2667 | 0.1959 | 0.0429 | 0.2232 | 0.2073 | 0.1860 | 0.3535 | 0.7681 | 0.3588 | 0.2984 | 0.3556 | 0.5613 | 0.7025 |
| no_correction | 0.2393 | 0.3170 | 0.2626 | 0.1963 | 0.0428 | 0.2181 | 0.3024 | 0.1458 | 0.3421 | 0.7216 | 0.3468 | 0.3829 | 0.4247 | 0.5939 | 0.6205 |

### Every cell, every bank

#### openai-clip-vit-b32-quickgelu__cifar100__sigma0.12 (sigma 0.12, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.3700 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2560 | 0.4000 | 0.6100 | 0.0780 | 0.5163 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.3700 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2560 | 0.4000 | 0.6100 | 0.0780 | 0.5163 |
| chowers_exact_projected_gap__coefficient_1 | 0.3700 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2560 | 0.4000 | 0.6100 | 0.0780 | 0.5163 |
| clean_boundary_active__step_0.0025 | 0.3700 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2520 | 0.4000 | 0.6080 | 0.0780 | 0.5163 |
| clean_boundary_active__step_0.005 | 0.3700 | 0.2620 | 0.1540 | 0.0000 | 0.2400 | 0.1280 | 0.2540 | 0.4000 | 0.6100 | 0.0740 | 0.5160 |
| clean_boundary_active__step_0.01 | 0.3740 | 0.2640 | 0.1560 | 0.0000 | 0.2420 | 0.1280 | 0.2540 | 0.4000 | 0.6100 | 0.0660 | 0.5196 |
| clean_boundary_active__step_0.02 | 0.3760 | 0.2680 | 0.1560 | 0.0000 | 0.2480 | 0.1280 | 0.2600 | 0.4020 | 0.6140 | 0.0560 | 0.5134 |
| clean_boundary_active__step_0.04 | 0.3760 | 0.2660 | 0.1580 | 0.0000 | 0.2440 | 0.1200 | 0.2620 | 0.4080 | 0.6160 | 0.0540 | 0.5097 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.3700 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2580 | 0.4000 | 0.6080 | 0.0780 | 0.5182 |
| cohen_aligned_noisy_margin__step_0.005 | 0.3700 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2620 | 0.4020 | 0.6080 | 0.0780 | 0.5217 |
| cohen_aligned_noisy_margin__step_0.01 | 0.3700 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2640 | 0.4020 | 0.6080 | 0.0780 | 0.5217 |
| cohen_aligned_noisy_margin__step_0.02 | 0.3740 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2520 | 0.4060 | 0.6100 | 0.0820 | 0.5218 |
| cohen_aligned_noisy_margin__step_0.04 | 0.3740 | 0.2600 | 0.1540 | 0.0000 | 0.2400 | 0.1280 | 0.2440 | 0.4080 | 0.6060 | 0.0820 | 0.5220 |
| control__learned_shared_translation_pure | 0.2180 | 0.1500 | 0.0900 | 0.0000 | 0.1360 | 0.3200 | 0.1480 | 0.2320 | 0.4300 | 0.2120 | 0.8104 |
| control__learned_shared_translation_tangent | 0.3680 | 0.2640 | 0.1500 | 0.0000 | 0.2480 | 0.1360 | 0.2040 | 0.3860 | 0.6160 | 0.0700 | 0.5721 |
| control__lowrank_tangent_r8 | 0.4600 | 0.3520 | 0.1940 | 0.0000 | 0.3200 | 0.0920 | 0.2000 | 0.5100 | 0.6420 | 0.0280 | 0.2939 |
| control__noisy_class_mean | 0.4360 | 0.3220 | 0.1820 | 0.0000 | 0.2420 | 0.1160 | 0.2180 | 0.4760 | 0.4540 | 0.0260 | 0.2961 |
| global_mean_centering__coefficient_0.25 | 0.3860 | 0.2660 | 0.1640 | 0.0000 | 0.2460 | 0.1280 | 0.2180 | 0.4000 | 0.6480 | 0.0760 | 0.5577 |
| global_mean_centering__coefficient_0.5 | 0.3460 | 0.2500 | 0.1500 | 0.0000 | 0.2380 | 0.1560 | 0.2260 | 0.3780 | 0.6020 | 0.1160 | 0.6305 |
| global_mean_centering__coefficient_1 | 0.2800 | 0.2060 | 0.1340 | 0.0000 | 0.1960 | 0.2060 | 0.1900 | 0.2940 | 0.4820 | 0.1740 | 0.7190 |
| gr_clip_style_two_sided__coefficient_1 | 0.3660 | 0.2800 | 0.1580 | 0.0000 | 0.2680 | 0.1500 | 0.2380 | 0.4080 | 0.6600 | 0.0640 | 0.5519 |
| no_correction | 0.3700 | 0.2600 | 0.1540 | 0.0000 | 0.2380 | 0.1280 | 0.2560 | 0.4000 | 0.6100 | 0.0780 | 0.5163 |

#### openai-clip-vit-b32-quickgelu__cifar100__sigma0.5 (sigma 0.5, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3140 | 0.2520 | 0.0680 | 0.6100 | 0.5640 | 0.9354 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3140 | 0.2520 | 0.0680 | 0.6100 | 0.5640 | 0.9354 |
| chowers_exact_projected_gap__coefficient_1 | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3140 | 0.2520 | 0.0680 | 0.6100 | 0.5640 | 0.9354 |
| clean_boundary_active__step_0.0025 | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3120 | 0.2540 | 0.0680 | 0.6080 | 0.5680 | 0.9364 |
| clean_boundary_active__step_0.005 | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3120 | 0.2520 | 0.0660 | 0.6100 | 0.5680 | 0.9369 |
| clean_boundary_active__step_0.01 | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3140 | 0.2560 | 0.0660 | 0.6100 | 0.5700 | 0.9369 |
| clean_boundary_active__step_0.02 | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3200 | 0.2580 | 0.0660 | 0.6140 | 0.5680 | 0.9380 |
| clean_boundary_active__step_0.04 | 0.0540 | 0.0460 | 0.0380 | 0.0300 | 0.0280 | 0.3240 | 0.2500 | 0.0700 | 0.6160 | 0.5740 | 0.9375 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3140 | 0.2520 | 0.0680 | 0.6100 | 0.5640 | 0.9354 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0540 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3100 | 0.2540 | 0.0680 | 0.6100 | 0.5640 | 0.9354 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0540 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3060 | 0.2540 | 0.0680 | 0.6040 | 0.5620 | 0.9352 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0540 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3000 | 0.2600 | 0.0700 | 0.6060 | 0.5600 | 0.9352 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0520 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.2920 | 0.2820 | 0.0700 | 0.6040 | 0.5540 | 0.9358 |
| control__learned_shared_translation_pure | 0.0240 | 0.0220 | 0.0200 | 0.0100 | 0.0100 | 0.1860 | 0.2300 | 0.0320 | 0.4180 | 0.3420 | 0.9606 |
| control__learned_shared_translation_tangent | 0.0560 | 0.0480 | 0.0420 | 0.0320 | 0.0300 | 0.3320 | 0.2500 | 0.0660 | 0.6160 | 0.5700 | 0.9377 |
| control__lowrank_tangent_r8 | 0.1040 | 0.0840 | 0.0700 | 0.0280 | 0.0240 | 0.0500 | 0.6060 | 0.1760 | 0.6240 | 0.0700 | 0.6062 |
| control__noisy_class_mean | 0.1100 | 0.0820 | 0.0700 | 0.0420 | 0.0080 | 0.0400 | 0.6260 | 0.1880 | 0.0800 | 0.0460 | 0.4429 |
| global_mean_centering__coefficient_0.25 | 0.0520 | 0.0480 | 0.0440 | 0.0280 | 0.0280 | 0.2960 | 0.2740 | 0.0680 | 0.6480 | 0.5320 | 0.9409 |
| global_mean_centering__coefficient_0.5 | 0.0560 | 0.0480 | 0.0360 | 0.0240 | 0.0240 | 0.2020 | 0.3420 | 0.0660 | 0.6020 | 0.3280 | 0.9414 |
| global_mean_centering__coefficient_1 | 0.0360 | 0.0320 | 0.0300 | 0.0300 | 0.0280 | 0.2100 | 0.2540 | 0.0460 | 0.4820 | 0.3740 | 0.9564 |
| gr_clip_style_two_sided__coefficient_1 | 0.0480 | 0.0440 | 0.0380 | 0.0300 | 0.0260 | 0.3140 | 0.2880 | 0.0700 | 0.6600 | 0.5780 | 0.9339 |
| no_correction | 0.0560 | 0.0460 | 0.0400 | 0.0300 | 0.0280 | 0.3140 | 0.2520 | 0.0680 | 0.6100 | 0.5640 | 0.9354 |

#### openai-clip-vit-b32-quickgelu__cifar10__sigma0.12 (sigma 0.12, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.7380 | 0.5840 | 0.3760 | 0.0000 | 0.5640 | 0.0600 | 0.0880 | 0.7660 | 0.9000 | 0.1520 | 0.1040 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.7380 | 0.5840 | 0.3760 | 0.0000 | 0.5640 | 0.0600 | 0.0880 | 0.7660 | 0.9000 | 0.1520 | 0.1040 |
| chowers_exact_projected_gap__coefficient_1 | 0.7380 | 0.5840 | 0.3760 | 0.0000 | 0.5640 | 0.0600 | 0.0880 | 0.7660 | 0.9000 | 0.1520 | 0.1040 |
| clean_boundary_active__step_0.0025 | 0.7400 | 0.5860 | 0.3760 | 0.0000 | 0.5660 | 0.0620 | 0.0880 | 0.7620 | 0.9000 | 0.1500 | 0.1012 |
| clean_boundary_active__step_0.005 | 0.7400 | 0.5860 | 0.3720 | 0.0000 | 0.5660 | 0.0600 | 0.0860 | 0.7640 | 0.9000 | 0.1500 | 0.1012 |
| clean_boundary_active__step_0.01 | 0.7400 | 0.5860 | 0.3740 | 0.0000 | 0.5660 | 0.0600 | 0.0940 | 0.7660 | 0.9000 | 0.1500 | 0.1004 |
| clean_boundary_active__step_0.02 | 0.7460 | 0.5900 | 0.3800 | 0.0000 | 0.5680 | 0.0560 | 0.0920 | 0.7680 | 0.8960 | 0.1480 | 0.0956 |
| clean_boundary_active__step_0.04 | 0.7600 | 0.5920 | 0.3820 | 0.0000 | 0.5640 | 0.0520 | 0.0800 | 0.7840 | 0.8920 | 0.1360 | 0.0896 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.7380 | 0.5840 | 0.3760 | 0.0000 | 0.5640 | 0.0600 | 0.0900 | 0.7660 | 0.9000 | 0.1500 | 0.1028 |
| cohen_aligned_noisy_margin__step_0.005 | 0.7380 | 0.5880 | 0.3760 | 0.0000 | 0.5680 | 0.0620 | 0.0900 | 0.7660 | 0.9000 | 0.1500 | 0.1024 |
| cohen_aligned_noisy_margin__step_0.01 | 0.7380 | 0.5880 | 0.3740 | 0.0000 | 0.5680 | 0.0620 | 0.0920 | 0.7640 | 0.9000 | 0.1500 | 0.1012 |
| cohen_aligned_noisy_margin__step_0.02 | 0.7420 | 0.5880 | 0.3760 | 0.0000 | 0.5700 | 0.0600 | 0.0920 | 0.7640 | 0.9020 | 0.1500 | 0.1012 |
| cohen_aligned_noisy_margin__step_0.04 | 0.7420 | 0.5920 | 0.3800 | 0.0000 | 0.5740 | 0.0600 | 0.0920 | 0.7720 | 0.9020 | 0.1420 | 0.0944 |
| control__learned_shared_translation_pure | 0.6720 | 0.5140 | 0.3560 | 0.0000 | 0.4980 | 0.0880 | 0.0820 | 0.7000 | 0.8920 | 0.2520 | 0.2304 |
| control__learned_shared_translation_tangent | 0.7720 | 0.6220 | 0.3900 | 0.0000 | 0.5960 | 0.0380 | 0.0800 | 0.8020 | 0.8920 | 0.1160 | 0.0560 |
| control__lowrank_tangent_r8 | 0.7500 | 0.6320 | 0.4640 | 0.0000 | 0.6140 | 0.0880 | 0.0500 | 0.7740 | 0.9120 | 0.1360 | 0.0976 |
| control__noisy_class_mean | 0.6660 | 0.5480 | 0.3500 | 0.0000 | 0.4720 | 0.1380 | 0.0640 | 0.6920 | 0.7600 | 0.1580 | 0.1472 |
| global_mean_centering__coefficient_0.25 | 0.7540 | 0.5980 | 0.3680 | 0.0000 | 0.5740 | 0.0520 | 0.0840 | 0.7780 | 0.8880 | 0.1220 | 0.0984 |
| global_mean_centering__coefficient_0.5 | 0.7260 | 0.5840 | 0.3640 | 0.0000 | 0.5680 | 0.0520 | 0.0720 | 0.7540 | 0.8620 | 0.1520 | 0.1512 |
| global_mean_centering__coefficient_1 | 0.6860 | 0.5600 | 0.3480 | 0.0000 | 0.5480 | 0.0700 | 0.0740 | 0.7040 | 0.8280 | 0.2040 | 0.2696 |
| gr_clip_style_two_sided__coefficient_1 | 0.7540 | 0.6040 | 0.3720 | 0.0000 | 0.5760 | 0.0620 | 0.0800 | 0.7740 | 0.8960 | 0.1580 | 0.1660 |
| no_correction | 0.7380 | 0.5840 | 0.3760 | 0.0000 | 0.5640 | 0.0600 | 0.0880 | 0.7660 | 0.9000 | 0.1520 | 0.1040 |

#### openai-clip-vit-b32-quickgelu__cifar10__sigma0.25 (sigma 0.25, 500 items, 10 classes, bridge, excluded from the primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.4120 | 0.3320 | 0.2420 | 0.1080 | 0.2380 | 0.0780 | 0.2020 | 0.4520 | 0.9000 | 0.3000 | 0.4176 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.4120 | 0.3320 | 0.2420 | 0.1080 | 0.2380 | 0.0780 | 0.2020 | 0.4520 | 0.9000 | 0.3000 | 0.4176 |
| chowers_exact_projected_gap__coefficient_1 | 0.4120 | 0.3320 | 0.2420 | 0.1080 | 0.2380 | 0.0780 | 0.2020 | 0.4520 | 0.9000 | 0.3000 | 0.4176 |
| clean_boundary_active__step_0.0025 | 0.4140 | 0.3360 | 0.2380 | 0.1080 | 0.2340 | 0.0800 | 0.2000 | 0.4560 | 0.9000 | 0.3020 | 0.4144 |
| clean_boundary_active__step_0.005 | 0.4140 | 0.3320 | 0.2420 | 0.1080 | 0.2380 | 0.0800 | 0.1980 | 0.4580 | 0.9000 | 0.3020 | 0.4136 |
| clean_boundary_active__step_0.01 | 0.4080 | 0.3320 | 0.2420 | 0.1100 | 0.2380 | 0.0780 | 0.2120 | 0.4520 | 0.9000 | 0.3060 | 0.4204 |
| clean_boundary_active__step_0.02 | 0.4060 | 0.3320 | 0.2480 | 0.1100 | 0.2440 | 0.0800 | 0.2080 | 0.4540 | 0.8960 | 0.3160 | 0.4280 |
| clean_boundary_active__step_0.04 | 0.4080 | 0.3360 | 0.2480 | 0.1140 | 0.2440 | 0.0860 | 0.2100 | 0.4600 | 0.8920 | 0.3300 | 0.4368 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.4140 | 0.3360 | 0.2420 | 0.1080 | 0.2380 | 0.0780 | 0.2000 | 0.4540 | 0.9000 | 0.3000 | 0.4168 |
| cohen_aligned_noisy_margin__step_0.005 | 0.4140 | 0.3360 | 0.2440 | 0.1080 | 0.2400 | 0.0800 | 0.2000 | 0.4560 | 0.9000 | 0.3000 | 0.4140 |
| cohen_aligned_noisy_margin__step_0.01 | 0.4140 | 0.3340 | 0.2460 | 0.1080 | 0.2420 | 0.0780 | 0.2020 | 0.4580 | 0.9000 | 0.3000 | 0.4132 |
| cohen_aligned_noisy_margin__step_0.02 | 0.4120 | 0.3360 | 0.2500 | 0.1100 | 0.2460 | 0.0780 | 0.2040 | 0.4560 | 0.9000 | 0.3040 | 0.4148 |
| cohen_aligned_noisy_margin__step_0.04 | 0.4100 | 0.3400 | 0.2500 | 0.1120 | 0.2460 | 0.0780 | 0.2080 | 0.4560 | 0.8980 | 0.3060 | 0.4176 |
| control__learned_shared_translation_pure | 0.3340 | 0.2640 | 0.1980 | 0.0900 | 0.1940 | 0.2080 | 0.1440 | 0.3700 | 0.8820 | 0.5260 | 0.5960 |
| control__learned_shared_translation_tangent | 0.4860 | 0.3940 | 0.2720 | 0.1200 | 0.2640 | 0.0720 | 0.1840 | 0.5360 | 0.8940 | 0.2580 | 0.2952 |
| control__lowrank_tangent_r8 | 0.5820 | 0.5040 | 0.4000 | 0.1920 | 0.3920 | 0.1020 | 0.1200 | 0.6320 | 0.9000 | 0.1220 | 0.0700 |
| control__noisy_class_mean | 0.4800 | 0.4000 | 0.3120 | 0.1560 | 0.2200 | 0.1700 | 0.1220 | 0.5140 | 0.5120 | 0.1600 | 0.1488 |
| global_mean_centering__coefficient_0.25 | 0.3980 | 0.3240 | 0.2400 | 0.1120 | 0.2380 | 0.0960 | 0.1800 | 0.4420 | 0.8880 | 0.2960 | 0.4664 |
| global_mean_centering__coefficient_0.5 | 0.4020 | 0.3240 | 0.2300 | 0.1200 | 0.2260 | 0.1400 | 0.1580 | 0.4400 | 0.8620 | 0.3960 | 0.5380 |
| global_mean_centering__coefficient_1 | 0.3720 | 0.2820 | 0.2300 | 0.1400 | 0.2260 | 0.2480 | 0.1340 | 0.4140 | 0.8280 | 0.5080 | 0.5972 |
| gr_clip_style_two_sided__coefficient_1 | 0.4700 | 0.3560 | 0.2660 | 0.1040 | 0.2620 | 0.0500 | 0.2280 | 0.5360 | 0.8960 | 0.1720 | 0.2676 |
| no_correction | 0.4120 | 0.3320 | 0.2420 | 0.1080 | 0.2380 | 0.0780 | 0.2020 | 0.4520 | 0.9000 | 0.3000 | 0.4176 |

#### openai-clip-vit-b32-quickgelu__cifar10__sigma0.5 (sigma 0.5, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1780 | 0.1500 | 0.1220 | 0.0720 | 0.0700 | 0.2780 | 0.2140 | 0.2300 | 0.9000 | 0.6980 | 0.6852 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1780 | 0.1500 | 0.1220 | 0.0720 | 0.0700 | 0.2780 | 0.2140 | 0.2300 | 0.9000 | 0.6980 | 0.6852 |
| chowers_exact_projected_gap__coefficient_1 | 0.1780 | 0.1500 | 0.1220 | 0.0720 | 0.0700 | 0.2780 | 0.2140 | 0.2300 | 0.9000 | 0.6980 | 0.6852 |
| clean_boundary_active__step_0.0025 | 0.1780 | 0.1480 | 0.1240 | 0.0680 | 0.0660 | 0.2800 | 0.2100 | 0.2320 | 0.9000 | 0.7000 | 0.6864 |
| clean_boundary_active__step_0.005 | 0.1800 | 0.1500 | 0.1240 | 0.0680 | 0.0660 | 0.2800 | 0.2120 | 0.2340 | 0.9000 | 0.7020 | 0.6928 |
| clean_boundary_active__step_0.01 | 0.1840 | 0.1500 | 0.1240 | 0.0680 | 0.0660 | 0.2880 | 0.2100 | 0.2340 | 0.9000 | 0.6980 | 0.6980 |
| clean_boundary_active__step_0.02 | 0.1880 | 0.1540 | 0.1240 | 0.0720 | 0.0680 | 0.2800 | 0.2040 | 0.2360 | 0.8960 | 0.7000 | 0.7024 |
| clean_boundary_active__step_0.04 | 0.1960 | 0.1640 | 0.1260 | 0.0800 | 0.0760 | 0.2680 | 0.2060 | 0.2520 | 0.8920 | 0.6880 | 0.7084 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1780 | 0.1500 | 0.1220 | 0.0680 | 0.0660 | 0.2800 | 0.2120 | 0.2340 | 0.9000 | 0.6940 | 0.6812 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1800 | 0.1520 | 0.1220 | 0.0680 | 0.0660 | 0.2800 | 0.2080 | 0.2360 | 0.9000 | 0.6960 | 0.6844 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1840 | 0.1480 | 0.1200 | 0.0700 | 0.0660 | 0.2740 | 0.2040 | 0.2380 | 0.9000 | 0.6920 | 0.6880 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1900 | 0.1520 | 0.1220 | 0.0700 | 0.0660 | 0.2580 | 0.2220 | 0.2420 | 0.8960 | 0.6840 | 0.6908 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1980 | 0.1580 | 0.1240 | 0.0780 | 0.0740 | 0.2440 | 0.2200 | 0.2540 | 0.8920 | 0.6640 | 0.6940 |
| control__learned_shared_translation_pure | 0.1580 | 0.1280 | 0.1040 | 0.0700 | 0.0680 | 0.3140 | 0.1420 | 0.1840 | 0.8880 | 0.7020 | 0.7776 |
| control__learned_shared_translation_tangent | 0.2240 | 0.1800 | 0.1440 | 0.0920 | 0.0880 | 0.1480 | 0.2800 | 0.2940 | 0.8920 | 0.5540 | 0.5900 |
| control__lowrank_tangent_r8 | 0.3640 | 0.3120 | 0.2680 | 0.1500 | 0.1460 | 0.0640 | 0.3020 | 0.4440 | 0.8940 | 0.1420 | 0.1020 |
| control__noisy_class_mean | 0.2840 | 0.2300 | 0.1900 | 0.1160 | 0.0520 | 0.1120 | 0.3300 | 0.3780 | 0.1980 | 0.1520 | 0.1900 |
| global_mean_centering__coefficient_0.25 | 0.1820 | 0.1580 | 0.1380 | 0.0900 | 0.0840 | 0.3960 | 0.1460 | 0.2100 | 0.8880 | 0.7720 | 0.7636 |
| global_mean_centering__coefficient_0.5 | 0.1740 | 0.1560 | 0.1360 | 0.1080 | 0.1020 | 0.4760 | 0.1000 | 0.1940 | 0.8620 | 0.8060 | 0.7832 |
| global_mean_centering__coefficient_1 | 0.1600 | 0.1500 | 0.1380 | 0.1080 | 0.1040 | 0.5460 | 0.0900 | 0.1800 | 0.8280 | 0.8380 | 0.8184 |
| gr_clip_style_two_sided__coefficient_1 | 0.1980 | 0.1600 | 0.1260 | 0.0580 | 0.0540 | 0.1240 | 0.3260 | 0.2760 | 0.8960 | 0.4280 | 0.5468 |
| no_correction | 0.1780 | 0.1500 | 0.1220 | 0.0720 | 0.0700 | 0.2780 | 0.2140 | 0.2300 | 0.9000 | 0.6980 | 0.6852 |

#### openai-clip-vit-b32-quickgelu__eurosat__sigma0.12 (sigma 0.12, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5267 | 0.0600 | 0.0933 | 0.4733 | 0.5000 | 0.7427 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5267 | 0.0600 | 0.0933 | 0.4733 | 0.5000 | 0.7427 |
| chowers_exact_projected_gap__coefficient_1 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5267 | 0.0600 | 0.0933 | 0.4733 | 0.5000 | 0.7427 |
| clean_boundary_active__step_0.0025 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5267 | 0.0600 | 0.0933 | 0.4733 | 0.5000 | 0.7427 |
| clean_boundary_active__step_0.005 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5267 | 0.0600 | 0.0933 | 0.4733 | 0.5000 | 0.7427 |
| clean_boundary_active__step_0.01 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5333 | 0.0667 | 0.0933 | 0.4800 | 0.5000 | 0.7440 |
| clean_boundary_active__step_0.02 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0333 | 0.5267 | 0.0800 | 0.1000 | 0.4667 | 0.4933 | 0.7467 |
| clean_boundary_active__step_0.04 | 0.0933 | 0.0533 | 0.0267 | 0.0000 | 0.0333 | 0.5133 | 0.0800 | 0.1000 | 0.4600 | 0.4800 | 0.7427 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5267 | 0.0733 | 0.0933 | 0.4733 | 0.5000 | 0.7427 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5200 | 0.0733 | 0.0933 | 0.4733 | 0.5000 | 0.7427 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0867 | 0.0600 | 0.0267 | 0.0000 | 0.0400 | 0.5200 | 0.0667 | 0.0933 | 0.4667 | 0.5000 | 0.7427 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0933 | 0.0533 | 0.0267 | 0.0000 | 0.0333 | 0.5467 | 0.0733 | 0.1000 | 0.4533 | 0.5133 | 0.7427 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0933 | 0.0600 | 0.0267 | 0.0000 | 0.0400 | 0.5733 | 0.0800 | 0.1133 | 0.4200 | 0.5400 | 0.7560 |
| control__learned_shared_translation_pure | 0.1000 | 0.0867 | 0.0467 | 0.0000 | 0.0733 | 0.6867 | 0.0467 | 0.1200 | 0.3600 | 0.7200 | 0.8267 |
| control__learned_shared_translation_tangent | 0.1867 | 0.0867 | 0.0067 | 0.0000 | 0.0800 | 0.2133 | 0.2467 | 0.2400 | 0.5267 | 0.3533 | 0.5733 |
| control__lowrank_tangent_r8 | 0.7067 | 0.5467 | 0.3867 | 0.0000 | 0.5067 | 0.1200 | 0.0467 | 0.7200 | 0.7533 | 0.1267 | 0.1173 |
| control__noisy_class_mean | 0.5667 | 0.4333 | 0.2267 | 0.0000 | 0.1467 | 0.1733 | 0.1067 | 0.6200 | 0.1733 | 0.1467 | 0.1933 |
| global_mean_centering__coefficient_0.25 | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.4733 | 0.1133 | 0.1067 | 0.5067 | 0.4733 | 0.7107 |
| global_mean_centering__coefficient_0.5 | 0.1000 | 0.0533 | 0.0133 | 0.0000 | 0.0467 | 0.2400 | 0.1333 | 0.1000 | 0.4600 | 0.2933 | 0.5907 |
| global_mean_centering__coefficient_1 | 0.1000 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5800 | 0.0800 | 0.1067 | 0.3000 | 0.5933 | 0.8040 |
| gr_clip_style_two_sided__coefficient_1 | 0.1600 | 0.0667 | 0.0000 | 0.0000 | 0.0467 | 0.2467 | 0.2467 | 0.2067 | 0.5400 | 0.3800 | 0.5560 |
| no_correction | 0.0867 | 0.0533 | 0.0267 | 0.0000 | 0.0400 | 0.5267 | 0.0600 | 0.0933 | 0.4733 | 0.5000 | 0.7427 |

#### openai-clip-vit-b32-quickgelu__eurosat__sigma0.5 (sigma 0.5, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6933 | 0.0067 | 0.0933 | 0.4733 | 0.9733 | 0.8947 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6933 | 0.0067 | 0.0933 | 0.4733 | 0.9733 | 0.8947 |
| chowers_exact_projected_gap__coefficient_1 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6933 | 0.0067 | 0.0933 | 0.4733 | 0.9733 | 0.8947 |
| clean_boundary_active__step_0.0025 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6867 | 0.0067 | 0.0933 | 0.4733 | 0.9733 | 0.8947 |
| clean_boundary_active__step_0.005 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6867 | 0.0067 | 0.0933 | 0.4733 | 0.9733 | 0.8947 |
| clean_boundary_active__step_0.01 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6867 | 0.0067 | 0.0933 | 0.4800 | 0.9733 | 0.8947 |
| clean_boundary_active__step_0.02 | 0.0933 | 0.0933 | 0.0867 | 0.0800 | 0.0600 | 0.6800 | 0.0067 | 0.0933 | 0.4667 | 0.9733 | 0.8947 |
| clean_boundary_active__step_0.04 | 0.0933 | 0.0933 | 0.0867 | 0.0800 | 0.0600 | 0.6667 | 0.0067 | 0.0933 | 0.4600 | 0.9733 | 0.8947 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6933 | 0.0067 | 0.0933 | 0.4733 | 0.9733 | 0.8947 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6867 | 0.0067 | 0.0933 | 0.4733 | 0.9733 | 0.8947 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6800 | 0.0000 | 0.0933 | 0.4667 | 0.9667 | 0.8933 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0933 | 0.0933 | 0.0867 | 0.0800 | 0.0533 | 0.6533 | 0.0000 | 0.0933 | 0.4533 | 0.9667 | 0.8933 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0933 | 0.0933 | 0.0867 | 0.0733 | 0.0533 | 0.5867 | 0.0200 | 0.1000 | 0.4200 | 0.9600 | 0.8920 |
| control__learned_shared_translation_pure | 0.1000 | 0.0933 | 0.0933 | 0.0933 | 0.0800 | 0.8533 | 0.0067 | 0.1000 | 0.3933 | 0.9800 | 0.8960 |
| control__learned_shared_translation_tangent | 0.1400 | 0.1133 | 0.0533 | 0.0133 | 0.0000 | 0.0400 | 0.3333 | 0.1733 | 0.5067 | 0.4000 | 0.6533 |
| control__lowrank_tangent_r8 | 0.5067 | 0.4400 | 0.3933 | 0.2733 | 0.2400 | 0.0467 | 0.2733 | 0.5533 | 0.7733 | 0.1867 | 0.3093 |
| control__noisy_class_mean | 0.4867 | 0.4133 | 0.3467 | 0.2267 | 0.0933 | 0.1000 | 0.1667 | 0.5333 | 0.1000 | 0.1667 | 0.2253 |
| global_mean_centering__coefficient_0.25 | 0.0933 | 0.0867 | 0.0733 | 0.0600 | 0.0467 | 0.7800 | 0.0133 | 0.0933 | 0.5067 | 0.9733 | 0.8947 |
| global_mean_centering__coefficient_0.5 | 0.0867 | 0.0800 | 0.0733 | 0.0667 | 0.0267 | 0.7400 | 0.0133 | 0.0867 | 0.4600 | 0.8800 | 0.8760 |
| global_mean_centering__coefficient_1 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0867 | 0.9000 | 0.0000 | 0.1000 | 0.3000 | 1.0000 | 0.9000 |
| gr_clip_style_two_sided__coefficient_1 | 0.1133 | 0.1067 | 0.1067 | 0.1000 | 0.0267 | 0.2333 | 0.2133 | 0.1200 | 0.5400 | 0.6133 | 0.7520 |
| no_correction | 0.0933 | 0.0933 | 0.0867 | 0.0867 | 0.0600 | 0.6933 | 0.0067 | 0.0933 | 0.4733 | 0.9733 | 0.8947 |

#### openai-clip-vit-l14-quickgelu__cifar100__sigma0.12 (sigma 0.12, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.5960 | 0.4820 | 0.3740 | 0.0000 | 0.4320 | 0.0800 | 0.1620 | 0.6300 | 0.7240 | 0.0500 | 0.3444 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.5960 | 0.4820 | 0.3740 | 0.0000 | 0.4320 | 0.0800 | 0.1620 | 0.6300 | 0.7240 | 0.0500 | 0.3444 |
| chowers_exact_projected_gap__coefficient_1 | 0.5960 | 0.4820 | 0.3740 | 0.0000 | 0.4320 | 0.0800 | 0.1620 | 0.6300 | 0.7240 | 0.0500 | 0.3444 |
| clean_boundary_active__step_0.0025 | 0.5960 | 0.4820 | 0.3740 | 0.0000 | 0.4320 | 0.0820 | 0.1620 | 0.6280 | 0.7240 | 0.0500 | 0.3466 |
| clean_boundary_active__step_0.005 | 0.5940 | 0.4820 | 0.3760 | 0.0000 | 0.4320 | 0.0820 | 0.1640 | 0.6280 | 0.7240 | 0.0500 | 0.3466 |
| clean_boundary_active__step_0.01 | 0.5940 | 0.4820 | 0.3760 | 0.0000 | 0.4320 | 0.0820 | 0.1680 | 0.6280 | 0.7220 | 0.0500 | 0.3466 |
| clean_boundary_active__step_0.02 | 0.5960 | 0.4800 | 0.3740 | 0.0000 | 0.4340 | 0.0900 | 0.1720 | 0.6280 | 0.7220 | 0.0500 | 0.3464 |
| clean_boundary_active__step_0.04 | 0.5940 | 0.4820 | 0.3740 | 0.0000 | 0.4360 | 0.0940 | 0.1660 | 0.6280 | 0.7220 | 0.0520 | 0.3457 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.5960 | 0.4820 | 0.3740 | 0.0000 | 0.4320 | 0.0800 | 0.1620 | 0.6300 | 0.7240 | 0.0500 | 0.3444 |
| cohen_aligned_noisy_margin__step_0.005 | 0.5960 | 0.4820 | 0.3740 | 0.0000 | 0.4320 | 0.0820 | 0.1640 | 0.6280 | 0.7240 | 0.0500 | 0.3464 |
| cohen_aligned_noisy_margin__step_0.01 | 0.5960 | 0.4820 | 0.3740 | 0.0000 | 0.4320 | 0.0820 | 0.1660 | 0.6260 | 0.7240 | 0.0500 | 0.3461 |
| cohen_aligned_noisy_margin__step_0.02 | 0.5920 | 0.4820 | 0.3760 | 0.0000 | 0.4320 | 0.0820 | 0.1700 | 0.6260 | 0.7240 | 0.0480 | 0.3459 |
| cohen_aligned_noisy_margin__step_0.04 | 0.5920 | 0.4840 | 0.3760 | 0.0000 | 0.4340 | 0.0860 | 0.1720 | 0.6260 | 0.7220 | 0.0480 | 0.3459 |
| control__learned_shared_translation_pure | 0.4740 | 0.3660 | 0.2660 | 0.0000 | 0.2360 | 0.1880 | 0.1400 | 0.5060 | 0.4120 | 0.1980 | 0.5894 |
| control__learned_shared_translation_tangent | 0.5620 | 0.4540 | 0.3380 | 0.0000 | 0.4380 | 0.1020 | 0.1280 | 0.5860 | 0.7600 | 0.0700 | 0.3889 |
| control__lowrank_tangent_r8 | 0.6540 | 0.5460 | 0.4100 | 0.0000 | 0.5140 | 0.0900 | 0.1280 | 0.6940 | 0.8020 | 0.0260 | 0.2276 |
| control__noisy_class_mean | 0.6100 | 0.5120 | 0.3540 | 0.0000 | 0.3840 | 0.1260 | 0.1320 | 0.6420 | 0.6160 | 0.0280 | 0.2483 |
| global_mean_centering__coefficient_0.25 | 0.5740 | 0.4800 | 0.3580 | 0.0000 | 0.4520 | 0.1000 | 0.1560 | 0.6080 | 0.7560 | 0.0640 | 0.3664 |
| global_mean_centering__coefficient_0.5 | 0.5480 | 0.4420 | 0.3360 | 0.0000 | 0.4300 | 0.1080 | 0.1420 | 0.5640 | 0.7440 | 0.0620 | 0.4355 |
| global_mean_centering__coefficient_1 | 0.4700 | 0.3900 | 0.2840 | 0.0000 | 0.3660 | 0.1640 | 0.1680 | 0.4960 | 0.6040 | 0.0600 | 0.5428 |
| gr_clip_style_two_sided__coefficient_1 | 0.5720 | 0.4800 | 0.3680 | 0.0000 | 0.4740 | 0.1000 | 0.1540 | 0.5980 | 0.7800 | 0.0340 | 0.3704 |
| no_correction | 0.5960 | 0.4820 | 0.3740 | 0.0000 | 0.4320 | 0.0800 | 0.1620 | 0.6300 | 0.7240 | 0.0500 | 0.3444 |

#### openai-clip-vit-l14-quickgelu__cifar100__sigma0.5 (sigma 0.5, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1260 | 0.1080 | 0.0940 | 0.0680 | 0.0600 | 0.1200 | 0.3620 | 0.1460 | 0.7240 | 0.1080 | 0.7964 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1260 | 0.1080 | 0.0940 | 0.0680 | 0.0600 | 0.1200 | 0.3620 | 0.1460 | 0.7240 | 0.1080 | 0.7964 |
| chowers_exact_projected_gap__coefficient_1 | 0.1260 | 0.1080 | 0.0940 | 0.0680 | 0.0600 | 0.1200 | 0.3620 | 0.1460 | 0.7240 | 0.1080 | 0.7964 |
| clean_boundary_active__step_0.0025 | 0.1260 | 0.1080 | 0.0940 | 0.0680 | 0.0600 | 0.1180 | 0.3640 | 0.1460 | 0.7240 | 0.1120 | 0.7972 |
| clean_boundary_active__step_0.005 | 0.1260 | 0.1080 | 0.0940 | 0.0680 | 0.0600 | 0.1180 | 0.3660 | 0.1460 | 0.7240 | 0.1140 | 0.7974 |
| clean_boundary_active__step_0.01 | 0.1240 | 0.1080 | 0.0940 | 0.0680 | 0.0600 | 0.1160 | 0.3680 | 0.1460 | 0.7220 | 0.1140 | 0.7976 |
| clean_boundary_active__step_0.02 | 0.1220 | 0.1080 | 0.0940 | 0.0640 | 0.0560 | 0.1180 | 0.3760 | 0.1460 | 0.7220 | 0.1140 | 0.7974 |
| clean_boundary_active__step_0.04 | 0.1220 | 0.1100 | 0.0960 | 0.0640 | 0.0560 | 0.1180 | 0.3820 | 0.1400 | 0.7220 | 0.1300 | 0.8002 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1260 | 0.1080 | 0.0940 | 0.0680 | 0.0600 | 0.1200 | 0.3640 | 0.1480 | 0.7240 | 0.1080 | 0.7954 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1260 | 0.1100 | 0.0940 | 0.0660 | 0.0600 | 0.1200 | 0.3720 | 0.1500 | 0.7200 | 0.1080 | 0.7933 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1280 | 0.1100 | 0.0940 | 0.0660 | 0.0600 | 0.1160 | 0.3780 | 0.1520 | 0.7200 | 0.1040 | 0.7929 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1280 | 0.1100 | 0.0940 | 0.0660 | 0.0600 | 0.1120 | 0.3800 | 0.1520 | 0.7200 | 0.1020 | 0.7924 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1260 | 0.1100 | 0.0920 | 0.0640 | 0.0580 | 0.1140 | 0.3880 | 0.1520 | 0.7240 | 0.1020 | 0.7916 |
| control__learned_shared_translation_pure | 0.0420 | 0.0360 | 0.0320 | 0.0240 | 0.0180 | 0.6280 | 0.0740 | 0.0480 | 0.3860 | 0.8320 | 0.9716 |
| control__learned_shared_translation_tangent | 0.1260 | 0.1060 | 0.0920 | 0.0640 | 0.0600 | 0.1000 | 0.4040 | 0.1460 | 0.7580 | 0.1160 | 0.7931 |
| control__lowrank_tangent_r8 | 0.2020 | 0.1720 | 0.1320 | 0.0940 | 0.0780 | 0.0880 | 0.3980 | 0.2400 | 0.7680 | 0.0600 | 0.5376 |
| control__noisy_class_mean | 0.1700 | 0.1500 | 0.1260 | 0.0860 | 0.0180 | 0.0940 | 0.4460 | 0.2320 | 0.1160 | 0.0540 | 0.3896 |
| global_mean_centering__coefficient_0.25 | 0.1180 | 0.1060 | 0.0920 | 0.0680 | 0.0640 | 0.1240 | 0.3940 | 0.1420 | 0.7560 | 0.1300 | 0.8100 |
| global_mean_centering__coefficient_0.5 | 0.1060 | 0.0880 | 0.0780 | 0.0720 | 0.0700 | 0.1600 | 0.3840 | 0.1300 | 0.7440 | 0.1740 | 0.8414 |
| global_mean_centering__coefficient_1 | 0.0900 | 0.0760 | 0.0660 | 0.0460 | 0.0440 | 0.2080 | 0.2900 | 0.1020 | 0.6040 | 0.1940 | 0.8936 |
| gr_clip_style_two_sided__coefficient_1 | 0.1140 | 0.1000 | 0.0860 | 0.0560 | 0.0560 | 0.1840 | 0.3000 | 0.1360 | 0.7800 | 0.3120 | 0.8413 |
| no_correction | 0.1260 | 0.1080 | 0.0940 | 0.0680 | 0.0600 | 0.1200 | 0.3620 | 0.1460 | 0.7240 | 0.1080 | 0.7964 |

#### openai-clip-vit-l14-quickgelu__cifar10__sigma0.12 (sigma 0.12, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| chowers_exact_projected_gap__coefficient_1 | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| clean_boundary_active__step_0.0025 | 0.8780 | 0.8200 | 0.6520 | 0.0000 | 0.8180 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| clean_boundary_active__step_0.005 | 0.8780 | 0.8200 | 0.6520 | 0.0000 | 0.8180 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| clean_boundary_active__step_0.01 | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| clean_boundary_active__step_0.02 | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0460 | 0.8980 | 0.9620 | 0.1180 | 0.0616 |
| clean_boundary_active__step_0.04 | 0.8800 | 0.8120 | 0.6540 | 0.0000 | 0.8100 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1180 | 0.0616 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| cohen_aligned_noisy_margin__step_0.005 | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| cohen_aligned_noisy_margin__step_0.01 | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| cohen_aligned_noisy_margin__step_0.02 | 0.8780 | 0.8220 | 0.6520 | 0.0000 | 0.8200 | 0.0240 | 0.0460 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |
| cohen_aligned_noisy_margin__step_0.04 | 0.8780 | 0.8200 | 0.6520 | 0.0000 | 0.8180 | 0.0240 | 0.0440 | 0.8980 | 0.9660 | 0.1180 | 0.0596 |
| control__learned_shared_translation_pure | 0.8620 | 0.7760 | 0.6340 | 0.0000 | 0.7700 | 0.0300 | 0.0420 | 0.8760 | 0.9560 | 0.1260 | 0.0840 |
| control__learned_shared_translation_tangent | 0.8880 | 0.8180 | 0.6580 | 0.0000 | 0.8160 | 0.0280 | 0.0320 | 0.9020 | 0.9700 | 0.1220 | 0.0592 |
| control__lowrank_tangent_r8 | 0.9100 | 0.8360 | 0.7020 | 0.0000 | 0.8300 | 0.0340 | 0.0180 | 0.9180 | 0.9720 | 0.1300 | 0.0736 |
| control__noisy_class_mean | 0.8540 | 0.7300 | 0.6060 | 0.0000 | 0.6520 | 0.0640 | 0.0140 | 0.8580 | 0.8040 | 0.1440 | 0.1184 |
| global_mean_centering__coefficient_0.25 | 0.8740 | 0.8040 | 0.6420 | 0.0000 | 0.8020 | 0.0240 | 0.0360 | 0.8900 | 0.9640 | 0.1220 | 0.0708 |
| global_mean_centering__coefficient_0.5 | 0.8640 | 0.7780 | 0.6400 | 0.0000 | 0.7760 | 0.0320 | 0.0380 | 0.8820 | 0.9600 | 0.1300 | 0.1056 |
| global_mean_centering__coefficient_1 | 0.8300 | 0.7500 | 0.6000 | 0.0000 | 0.7360 | 0.0480 | 0.0480 | 0.8480 | 0.9300 | 0.1620 | 0.1612 |
| gr_clip_style_two_sided__coefficient_1 | 0.9000 | 0.8040 | 0.6520 | 0.0000 | 0.8020 | 0.0340 | 0.0220 | 0.9060 | 0.9660 | 0.1420 | 0.0852 |
| no_correction | 0.8780 | 0.8180 | 0.6520 | 0.0000 | 0.8160 | 0.0240 | 0.0440 | 0.9000 | 0.9620 | 0.1160 | 0.0560 |

#### openai-clip-vit-l14-quickgelu__cifar10__sigma0.25 (sigma 0.25, 500 items, 10 classes, bridge, excluded from the primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.6760 | 0.6120 | 0.5100 | 0.3200 | 0.5060 | 0.0480 | 0.1240 | 0.7160 | 0.9620 | 0.1620 | 0.1728 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.6760 | 0.6120 | 0.5100 | 0.3200 | 0.5060 | 0.0480 | 0.1240 | 0.7160 | 0.9620 | 0.1620 | 0.1728 |
| chowers_exact_projected_gap__coefficient_1 | 0.6760 | 0.6120 | 0.5100 | 0.3200 | 0.5060 | 0.0480 | 0.1240 | 0.7160 | 0.9620 | 0.1620 | 0.1728 |
| clean_boundary_active__step_0.0025 | 0.6760 | 0.6120 | 0.5100 | 0.3180 | 0.5060 | 0.0500 | 0.1240 | 0.7160 | 0.9620 | 0.1620 | 0.1728 |
| clean_boundary_active__step_0.005 | 0.6760 | 0.6140 | 0.5080 | 0.3200 | 0.5040 | 0.0500 | 0.1220 | 0.7160 | 0.9620 | 0.1620 | 0.1732 |
| clean_boundary_active__step_0.01 | 0.6780 | 0.6100 | 0.5080 | 0.3200 | 0.5040 | 0.0520 | 0.1220 | 0.7140 | 0.9620 | 0.1620 | 0.1728 |
| clean_boundary_active__step_0.02 | 0.6780 | 0.6120 | 0.5100 | 0.3200 | 0.5060 | 0.0500 | 0.1200 | 0.7100 | 0.9620 | 0.1620 | 0.1708 |
| clean_boundary_active__step_0.04 | 0.6760 | 0.6080 | 0.5080 | 0.3180 | 0.5040 | 0.0500 | 0.1280 | 0.7100 | 0.9620 | 0.1600 | 0.1712 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.6740 | 0.6140 | 0.5080 | 0.3200 | 0.5040 | 0.0500 | 0.1260 | 0.7180 | 0.9620 | 0.1620 | 0.1736 |
| cohen_aligned_noisy_margin__step_0.005 | 0.6740 | 0.6140 | 0.5080 | 0.3200 | 0.5040 | 0.0480 | 0.1320 | 0.7180 | 0.9620 | 0.1620 | 0.1736 |
| cohen_aligned_noisy_margin__step_0.01 | 0.6760 | 0.6140 | 0.5080 | 0.3220 | 0.5040 | 0.0480 | 0.1300 | 0.7200 | 0.9620 | 0.1620 | 0.1716 |
| cohen_aligned_noisy_margin__step_0.02 | 0.6780 | 0.6160 | 0.5080 | 0.3240 | 0.5040 | 0.0480 | 0.1260 | 0.7180 | 0.9640 | 0.1640 | 0.1720 |
| cohen_aligned_noisy_margin__step_0.04 | 0.6800 | 0.6160 | 0.5100 | 0.3300 | 0.5060 | 0.0460 | 0.1260 | 0.7240 | 0.9660 | 0.1600 | 0.1672 |
| control__learned_shared_translation_pure | 0.6180 | 0.5600 | 0.4620 | 0.3000 | 0.4520 | 0.0740 | 0.1140 | 0.6260 | 0.9540 | 0.2200 | 0.3028 |
| control__learned_shared_translation_tangent | 0.7180 | 0.6200 | 0.5140 | 0.3480 | 0.5100 | 0.0440 | 0.0940 | 0.7500 | 0.9700 | 0.1700 | 0.1328 |
| control__lowrank_tangent_r8 | 0.7480 | 0.6680 | 0.5880 | 0.4160 | 0.5860 | 0.0860 | 0.0580 | 0.7660 | 0.9660 | 0.1500 | 0.1160 |
| control__noisy_class_mean | 0.6480 | 0.5660 | 0.4920 | 0.3480 | 0.4340 | 0.1160 | 0.0840 | 0.6700 | 0.8320 | 0.1640 | 0.1652 |
| global_mean_centering__coefficient_0.25 | 0.6920 | 0.6000 | 0.4940 | 0.3260 | 0.4900 | 0.0540 | 0.0960 | 0.7140 | 0.9640 | 0.1520 | 0.1828 |
| global_mean_centering__coefficient_0.5 | 0.6720 | 0.5880 | 0.4740 | 0.3180 | 0.4740 | 0.0680 | 0.0920 | 0.6940 | 0.9600 | 0.2140 | 0.2448 |
| global_mean_centering__coefficient_1 | 0.6260 | 0.5380 | 0.4420 | 0.2940 | 0.4420 | 0.0980 | 0.0920 | 0.6580 | 0.9300 | 0.2600 | 0.3324 |
| gr_clip_style_two_sided__coefficient_1 | 0.6900 | 0.6140 | 0.5020 | 0.3300 | 0.4980 | 0.0560 | 0.1080 | 0.7280 | 0.9660 | 0.1700 | 0.1804 |
| no_correction | 0.6760 | 0.6120 | 0.5100 | 0.3200 | 0.5060 | 0.0480 | 0.1240 | 0.7160 | 0.9620 | 0.1620 | 0.1728 |

#### openai-clip-vit-l14-quickgelu__cifar10__sigma0.5 (sigma 0.5, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.3420 | 0.2960 | 0.2300 | 0.1640 | 0.1620 | 0.0920 | 0.2380 | 0.4120 | 0.9620 | 0.2640 | 0.4144 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.3420 | 0.2960 | 0.2300 | 0.1640 | 0.1620 | 0.0920 | 0.2380 | 0.4120 | 0.9620 | 0.2640 | 0.4144 |
| chowers_exact_projected_gap__coefficient_1 | 0.3420 | 0.2960 | 0.2300 | 0.1640 | 0.1620 | 0.0920 | 0.2380 | 0.4120 | 0.9620 | 0.2640 | 0.4144 |
| clean_boundary_active__step_0.0025 | 0.3420 | 0.2960 | 0.2300 | 0.1640 | 0.1620 | 0.0920 | 0.2380 | 0.4120 | 0.9620 | 0.2640 | 0.4144 |
| clean_boundary_active__step_0.005 | 0.3420 | 0.3000 | 0.2300 | 0.1640 | 0.1620 | 0.0960 | 0.2360 | 0.4080 | 0.9620 | 0.2700 | 0.4196 |
| clean_boundary_active__step_0.01 | 0.3460 | 0.3020 | 0.2260 | 0.1640 | 0.1620 | 0.0960 | 0.2260 | 0.4060 | 0.9620 | 0.2700 | 0.4196 |
| clean_boundary_active__step_0.02 | 0.3420 | 0.3060 | 0.2300 | 0.1640 | 0.1620 | 0.1020 | 0.2300 | 0.4000 | 0.9620 | 0.2720 | 0.4256 |
| clean_boundary_active__step_0.04 | 0.3440 | 0.3000 | 0.2340 | 0.1660 | 0.1640 | 0.1040 | 0.2300 | 0.4000 | 0.9620 | 0.2680 | 0.4352 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.3420 | 0.2980 | 0.2340 | 0.1640 | 0.1620 | 0.0920 | 0.2380 | 0.4100 | 0.9620 | 0.2600 | 0.4124 |
| cohen_aligned_noisy_margin__step_0.005 | 0.3480 | 0.2980 | 0.2340 | 0.1640 | 0.1620 | 0.0960 | 0.2320 | 0.4080 | 0.9620 | 0.2560 | 0.4124 |
| cohen_aligned_noisy_margin__step_0.01 | 0.3500 | 0.2980 | 0.2360 | 0.1640 | 0.1620 | 0.0940 | 0.2320 | 0.4140 | 0.9620 | 0.2460 | 0.4064 |
| cohen_aligned_noisy_margin__step_0.02 | 0.3540 | 0.3040 | 0.2400 | 0.1660 | 0.1640 | 0.0900 | 0.2300 | 0.4180 | 0.9660 | 0.2340 | 0.3972 |
| cohen_aligned_noisy_margin__step_0.04 | 0.3680 | 0.3160 | 0.2580 | 0.1680 | 0.1660 | 0.0800 | 0.2240 | 0.4160 | 0.9660 | 0.2200 | 0.3784 |
| control__learned_shared_translation_pure | 0.2420 | 0.2220 | 0.1980 | 0.1300 | 0.1280 | 0.1920 | 0.1860 | 0.2860 | 0.9620 | 0.4780 | 0.6360 |
| control__learned_shared_translation_tangent | 0.3820 | 0.3340 | 0.2740 | 0.2160 | 0.2160 | 0.0680 | 0.2920 | 0.4400 | 0.9700 | 0.2140 | 0.2660 |
| control__lowrank_tangent_r8 | 0.4560 | 0.4120 | 0.3540 | 0.2780 | 0.2740 | 0.1040 | 0.1780 | 0.5060 | 0.9640 | 0.1380 | 0.1164 |
| control__noisy_class_mean | 0.3960 | 0.3520 | 0.2920 | 0.2160 | 0.1020 | 0.1320 | 0.2040 | 0.4500 | 0.3040 | 0.1740 | 0.1876 |
| global_mean_centering__coefficient_0.25 | 0.3520 | 0.3080 | 0.2520 | 0.1540 | 0.1520 | 0.0860 | 0.2260 | 0.4020 | 0.9640 | 0.2520 | 0.4164 |
| global_mean_centering__coefficient_0.5 | 0.3540 | 0.3020 | 0.2500 | 0.1720 | 0.1720 | 0.1320 | 0.2000 | 0.3800 | 0.9600 | 0.3500 | 0.5224 |
| global_mean_centering__coefficient_1 | 0.3240 | 0.2940 | 0.2520 | 0.1740 | 0.1740 | 0.2120 | 0.1380 | 0.3520 | 0.9300 | 0.4180 | 0.6060 |
| gr_clip_style_two_sided__coefficient_1 | 0.3420 | 0.2880 | 0.2580 | 0.1840 | 0.1820 | 0.1400 | 0.2040 | 0.3940 | 0.9660 | 0.2380 | 0.4748 |
| no_correction | 0.3420 | 0.2960 | 0.2300 | 0.1640 | 0.1620 | 0.0920 | 0.2380 | 0.4120 | 0.9620 | 0.2640 | 0.4144 |

#### openai-clip-vit-l14-quickgelu__eurosat__sigma0.12 (sigma 0.12, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.2400 | 0.1600 | 0.1000 | 0.0000 | 0.1333 | 0.4200 | 0.0667 | 0.2667 | 0.6600 | 0.5933 | 0.7373 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.2400 | 0.1600 | 0.1000 | 0.0000 | 0.1333 | 0.4200 | 0.0667 | 0.2667 | 0.6600 | 0.5933 | 0.7373 |
| chowers_exact_projected_gap__coefficient_1 | 0.2400 | 0.1600 | 0.1000 | 0.0000 | 0.1333 | 0.4200 | 0.0667 | 0.2667 | 0.6600 | 0.5933 | 0.7373 |
| clean_boundary_active__step_0.0025 | 0.2400 | 0.1667 | 0.1000 | 0.0000 | 0.1333 | 0.4267 | 0.0733 | 0.2667 | 0.6600 | 0.5933 | 0.7373 |
| clean_boundary_active__step_0.005 | 0.2400 | 0.1667 | 0.1000 | 0.0000 | 0.1333 | 0.4267 | 0.0867 | 0.2800 | 0.6600 | 0.6133 | 0.7373 |
| clean_boundary_active__step_0.01 | 0.2467 | 0.1667 | 0.0867 | 0.0000 | 0.1267 | 0.4400 | 0.0800 | 0.2733 | 0.6600 | 0.6200 | 0.7347 |
| clean_boundary_active__step_0.02 | 0.2400 | 0.1733 | 0.0933 | 0.0000 | 0.1333 | 0.4533 | 0.0600 | 0.2667 | 0.6533 | 0.6200 | 0.7347 |
| clean_boundary_active__step_0.04 | 0.2267 | 0.1733 | 0.0933 | 0.0000 | 0.1333 | 0.4600 | 0.0667 | 0.2467 | 0.6400 | 0.6200 | 0.7307 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.2400 | 0.1600 | 0.1000 | 0.0000 | 0.1267 | 0.4267 | 0.0733 | 0.2667 | 0.6467 | 0.5933 | 0.7373 |
| cohen_aligned_noisy_margin__step_0.005 | 0.2400 | 0.1667 | 0.0933 | 0.0000 | 0.1267 | 0.4267 | 0.0733 | 0.2667 | 0.6467 | 0.5933 | 0.7373 |
| cohen_aligned_noisy_margin__step_0.01 | 0.2467 | 0.1667 | 0.0867 | 0.0000 | 0.1267 | 0.4333 | 0.0733 | 0.2800 | 0.6400 | 0.6200 | 0.7387 |
| cohen_aligned_noisy_margin__step_0.02 | 0.2533 | 0.1733 | 0.0933 | 0.0000 | 0.1333 | 0.4467 | 0.0600 | 0.2733 | 0.6333 | 0.6133 | 0.7307 |
| cohen_aligned_noisy_margin__step_0.04 | 0.2400 | 0.1867 | 0.0933 | 0.0000 | 0.1467 | 0.4733 | 0.0933 | 0.2600 | 0.6133 | 0.6133 | 0.7240 |
| control__learned_shared_translation_pure | 0.2200 | 0.1600 | 0.1067 | 0.0000 | 0.1600 | 0.4667 | 0.0400 | 0.2267 | 0.6467 | 0.6000 | 0.7827 |
| control__learned_shared_translation_tangent | 0.2733 | 0.1467 | 0.0533 | 0.0000 | 0.1333 | 0.2600 | 0.1067 | 0.2933 | 0.6600 | 0.4267 | 0.6013 |
| control__lowrank_tangent_r8 | 0.7467 | 0.6267 | 0.4933 | 0.0000 | 0.6133 | 0.1133 | 0.0200 | 0.7533 | 0.8933 | 0.1200 | 0.0760 |
| control__noisy_class_mean | 0.5800 | 0.4200 | 0.2733 | 0.0000 | 0.1067 | 0.1400 | 0.0867 | 0.6000 | 0.2067 | 0.2067 | 0.2453 |
| global_mean_centering__coefficient_0.25 | 0.2400 | 0.1600 | 0.0867 | 0.0000 | 0.1267 | 0.4133 | 0.0467 | 0.2600 | 0.6667 | 0.6267 | 0.7307 |
| global_mean_centering__coefficient_0.5 | 0.2133 | 0.1667 | 0.0800 | 0.0000 | 0.1600 | 0.4400 | 0.0533 | 0.2267 | 0.6133 | 0.5733 | 0.7480 |
| global_mean_centering__coefficient_1 | 0.1667 | 0.1200 | 0.0600 | 0.0000 | 0.1200 | 0.4867 | 0.0467 | 0.1867 | 0.4467 | 0.5933 | 0.7947 |
| gr_clip_style_two_sided__coefficient_1 | 0.2267 | 0.1600 | 0.0867 | 0.0000 | 0.1533 | 0.2933 | 0.1533 | 0.2333 | 0.7667 | 0.3467 | 0.5627 |
| no_correction | 0.2400 | 0.1600 | 0.1000 | 0.0000 | 0.1333 | 0.4200 | 0.0667 | 0.2667 | 0.6600 | 0.5933 | 0.7373 |

#### openai-clip-vit-l14-quickgelu__eurosat__sigma0.5 (sigma 0.5, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0133 | 0.8933 | 0.0000 | 0.1000 | 0.6600 | 1.0000 | 0.9000 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0133 | 0.8933 | 0.0000 | 0.1000 | 0.6600 | 1.0000 | 0.9000 |
| chowers_exact_projected_gap__coefficient_1 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0133 | 0.8933 | 0.0000 | 0.1000 | 0.6600 | 1.0000 | 0.9000 |
| clean_boundary_active__step_0.0025 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0133 | 0.8933 | 0.0000 | 0.1000 | 0.6600 | 1.0000 | 0.9000 |
| clean_boundary_active__step_0.005 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0133 | 0.8867 | 0.0000 | 0.1000 | 0.6600 | 1.0000 | 0.9000 |
| clean_boundary_active__step_0.01 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0067 | 0.8867 | 0.0000 | 0.1000 | 0.6600 | 1.0000 | 0.9000 |
| clean_boundary_active__step_0.02 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0067 | 0.8867 | 0.0000 | 0.1000 | 0.6533 | 1.0000 | 0.9000 |
| clean_boundary_active__step_0.04 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0000 | 0.8867 | 0.0000 | 0.1000 | 0.6400 | 1.0000 | 0.9000 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0067 | 0.8933 | 0.0000 | 0.1000 | 0.6467 | 1.0000 | 0.9000 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0067 | 0.8867 | 0.0000 | 0.1000 | 0.6467 | 1.0000 | 0.9000 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0067 | 0.8867 | 0.0000 | 0.1000 | 0.6400 | 1.0000 | 0.9000 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0067 | 0.8867 | 0.0000 | 0.1000 | 0.6333 | 1.0000 | 0.9000 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0000 | 0.8867 | 0.0000 | 0.1000 | 0.6200 | 1.0000 | 0.9000 |
| control__learned_shared_translation_pure | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.9000 | 0.0000 | 0.1000 | 0.5400 | 1.0000 | 0.9000 |
| control__learned_shared_translation_tangent | 0.0867 | 0.0800 | 0.0733 | 0.0600 | 0.0467 | 0.3067 | 0.2200 | 0.1667 | 0.6467 | 0.6267 | 0.7387 |
| control__lowrank_tangent_r8 | 0.5067 | 0.4600 | 0.3933 | 0.3200 | 0.3000 | 0.1067 | 0.1267 | 0.5200 | 0.8867 | 0.2133 | 0.3493 |
| control__noisy_class_mean | 0.4600 | 0.4467 | 0.4133 | 0.3133 | 0.0800 | 0.1867 | 0.1267 | 0.4867 | 0.1133 | 0.2000 | 0.2813 |
| global_mean_centering__coefficient_0.25 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0067 | 0.8867 | 0.0000 | 0.1000 | 0.6667 | 1.0000 | 0.9000 |
| global_mean_centering__coefficient_0.5 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0000 | 0.8867 | 0.0000 | 0.1000 | 0.6133 | 1.0000 | 0.9000 |
| global_mean_centering__coefficient_1 | 0.0733 | 0.0600 | 0.0467 | 0.0267 | 0.0133 | 0.4000 | 0.0667 | 0.0933 | 0.4467 | 0.6467 | 0.8293 |
| gr_clip_style_two_sided__coefficient_1 | 0.1200 | 0.1067 | 0.1000 | 0.0867 | 0.0133 | 0.6067 | 0.0067 | 0.1200 | 0.7667 | 0.9733 | 0.8947 |
| no_correction | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0133 | 0.8933 | 0.0000 | 0.1000 | 0.6600 | 1.0000 | 0.9000 |

### Comparison with the registered d1_07 outputs

Analysis directory `results/satml2027_local/analysis/EXP-20260920-019A`; tolerance 1e-09.

| output | rows compared | max abs deviation | within tolerance |
|---|---|---|---|
| cells_csv | 308 | 0.000e+00 | yes |
| concentration_csv | 308 | 0.000e+00 | yes |
| macro_csv | 22 | 0.000e+00 | yes |
| contrasts_json (point, lower, upper) | 21 | 5.551e-17 | yes |
| contrasts_json critical value | 1 | 2.776e-17 | yes |

d1_07 band metadata: {"confidence_level": 0.95, "critical_max_absolute_deviation": 0.01838888888888887, "method": "shared_item_multiplicity_dataset_class_stratified_max_absolute_deviation", "replicates": 100000, "seed": 2026091605}; seed match = True, replicates match = True

## EXP-20260920-019B

- registration `3fcdbade664a3ded421e724060ddf810fe8fbc47f37b1c6d582515dbe9a96c20` (recomputed; file `a112916a9c3c8c04d4995864100603c9b95fcf5bf57d60dfadb5afb54a32c193`; match = True)
- alpha 0.001, selection 128, confirmation 4096, k_min at r = sigma recomputed 3518 (registered 3518)
- primary cells (18): openai-clip-rn50-quickgelu__cifar100__sigma0.25, openai-clip-rn50-quickgelu__cifar100__sigma0.5, openai-clip-rn50-quickgelu__cifar10__sigma0.25, openai-clip-rn50-quickgelu__cifar10__sigma0.5, openai-clip-rn50-quickgelu__eurosat__sigma0.25, openai-clip-rn50-quickgelu__eurosat__sigma0.5, openai-clip-vit-b16-quickgelu__cifar100__sigma0.25, openai-clip-vit-b16-quickgelu__cifar100__sigma0.5, openai-clip-vit-b16-quickgelu__cifar10__sigma0.25, openai-clip-vit-b16-quickgelu__cifar10__sigma0.5, openai-clip-vit-b16-quickgelu__eurosat__sigma0.25, openai-clip-vit-b16-quickgelu__eurosat__sigma0.5, openclip-vit-b32-laion2b__cifar100__sigma0.25, openclip-vit-b32-laion2b__cifar100__sigma0.5, openclip-vit-b32-laion2b__cifar10__sigma0.25, openclip-vit-b32-laion2b__cifar10__sigma0.5, openclip-vit-b32-laion2b__eurosat__sigma0.25, openclip-vit-b32-laion2b__eurosat__sigma0.5
- bridge cells excluded from the primary (0): none
- registered band: numpy.random.default_rng (PCG64), seed 2026091605, 100000 replicates, 120 strata, 1150 unique identities, 6900 positions; critical max-abs deviation = 0.015630

### Primary estimand: equal-cell macro standard CA at r = sigma (reference no_correction = 0.0795)

| bank | macro CA@sigma | delta vs no_correction | lower | upper | contains 0 | excludes 0 (+) | excludes 0 (-) |
|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0795 | 0.0000 | -0.0156 | 0.0156 | yes | no | no |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0795 | 0.0000 | -0.0156 | 0.0156 | yes | no | no |
| chowers_exact_projected_gap__coefficient_1 | 0.0795 | 0.0000 | -0.0156 | 0.0156 | yes | no | no |
| clean_boundary_active__step_0.0025 | 0.0793 | -0.0002 | -0.0159 | 0.0154 | yes | no | no |
| clean_boundary_active__step_0.005 | 0.0793 | -0.0002 | -0.0158 | 0.0154 | yes | no | no |
| clean_boundary_active__step_0.01 | 0.0794 | -0.0001 | -0.0158 | 0.0155 | yes | no | no |
| clean_boundary_active__step_0.02 | 0.0780 | -0.0015 | -0.0171 | 0.0141 | yes | no | no |
| clean_boundary_active__step_0.04 | 0.0782 | -0.0013 | -0.0169 | 0.0143 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0799 | 0.0004 | -0.0153 | 0.0160 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.005 | 0.0798 | 0.0003 | -0.0154 | 0.0159 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.01 | 0.0794 | -0.0001 | -0.0157 | 0.0156 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.02 | 0.0800 | 0.0005 | -0.0151 | 0.0161 | yes | no | no |
| cohen_aligned_noisy_margin__step_0.04 | 0.0796 | 0.0001 | -0.0155 | 0.0157 | yes | no | no |
| control__learned_shared_translation_pure | 0.0787 | -0.0008 | -0.0164 | 0.0148 | yes | no | no |
| control__learned_shared_translation_tangent | 0.0891 | 0.0096 | -0.0060 | 0.0252 | yes | no | no |
| control__lowrank_tangent_r8 | 0.2163 | 0.1367 | 0.1211 | 0.1524 | no | yes | no |
| control__noisy_class_mean | 0.1529 | 0.0734 | 0.0578 | 0.0890 | no | yes | no |
| global_mean_centering__coefficient_0.25 | 0.0821 | 0.0026 | -0.0131 | 0.0182 | yes | no | no |
| global_mean_centering__coefficient_0.5 | 0.0870 | 0.0075 | -0.0081 | 0.0231 | yes | no | no |
| global_mean_centering__coefficient_1 | 0.0856 | 0.0061 | -0.0096 | 0.0217 | yes | no | no |
| gr_clip_style_two_sided__coefficient_1 | 0.0837 | 0.0042 | -0.0114 | 0.0198 | yes | no | no |

### Registered predictions (pre-registered wording)

- **X1** (falsifiable): "report every frozen-grid candidate's macro standard-CA change at r=sigma relative to no correction against the prospective practical-effect region, without an equivalence claim"
  - region: absolute CA 0.005 (reporting region only; not an equivalence margin or TOST); critical = 0.015629629629629632; band narrower than region = False
  - 15/17 point estimates inside the region; 0/17 intervals within the region; 17/17 contain zero; 0/17 exclude zero on the positive side; 0/17 on the negative side
  - outcome: reported: all 17 frozen-grid contrasts listed with point estimate and registered interval; no equivalence claim is made; the band is wider than the +-0.005 reporting region, so no 'inside the region' language is permitted
- **X3** (falsifiable): "the learned shared-translation control stays inside the same band"
  - control__learned_shared_translation_tangent: point +0.0096, interval [-0.0060, +0.0252], contains zero = True
  - companion control__learned_shared_translation_pure: point -0.0008, interval [-0.0164, +0.0148], contains zero = True
  - outcome: confirmed
- **X4** (falsifiable positive control): "the per-class control banks leave the band"
  - control__lowrank_tangent_r8: point +0.1367, interval [+0.1211, +0.1524], lower > 0 = True, upper < 0 = False
  - control__noisy_class_mean: point +0.0734, interval [+0.0578, +0.0890], lower > 0 = True, upper < 0 = False
  - outcome: confirmed
- **X5** (falsifiable): "top-class share and predicted-class Gini of the zero-shot bank increase with sigma"
  - rule: 'supported' only if both statistics increase in every consecutive sigma pair of every (model, dataset); otherwise 'not uniformly supported'; descriptive, no inferential statement
  - openai-clip-rn50-quickgelu|cifar100|0.25->0.5: top-class share 0.618 -> 0.618 (not up); Gini 0.950 -> 0.976 (up)
  - openai-clip-rn50-quickgelu|cifar10|0.25->0.5: top-class share 0.532 -> 0.506 (not up); Gini 0.749 -> 0.795 (up)
  - openai-clip-rn50-quickgelu|eurosat|0.25->0.5: top-class share 0.653 -> 0.500 (not up); Gini 0.812 -> 0.785 (not up)
  - openai-clip-vit-b16-quickgelu|cifar100|0.25->0.5: top-class share 0.544 -> 0.644 (up); Gini 0.865 -> 0.966 (up)
  - openai-clip-vit-b16-quickgelu|cifar10|0.25->0.5: top-class share 0.244 -> 0.350 (up); Gini 0.416 -> 0.622 (up)
  - openai-clip-vit-b16-quickgelu|eurosat|0.25->0.5: top-class share 0.807 -> 0.433 (not up); Gini 0.815 -> 0.743 (not up)
  - openclip-vit-b32-laion2b|cifar100|0.25->0.5: top-class share 0.570 -> 0.786 (up); Gini 0.904 -> 0.967 (up)
  - openclip-vit-b32-laion2b|cifar10|0.25->0.5: top-class share 0.298 -> 0.708 (up); Gini 0.505 -> 0.799 (up)
  - openclip-vit-b32-laion2b|eurosat|0.25->0.5: top-class share 0.707 -> 0.900 (up); Gini 0.803 -> 0.875 (up)
  - outcome: not uniformly supported

### Identity bank (no_correction) regime per cell

| cell | sigma | role | clean acc | smoothed acc | abstention | CA@sigma | cert-wrong@sigma | top-class share | Gini | classes predicted |
|---|---|---|---|---|---|---|---|---|---|---|
| openai-clip-rn50-quickgelu__cifar100__sigma0.25 | 0.2500 | primary | 0.3880 | 0.0380 | 0.2120 | 0.0100 | 0.4180 | 0.6180 | 0.9496 | 26 |
| openai-clip-rn50-quickgelu__cifar100__sigma0.5 | 0.5000 | primary | 0.3880 | 0.0100 | 0.0420 | 0.0100 | 0.6500 | 0.6180 | 0.9764 | 9 |
| openai-clip-rn50-quickgelu__cifar10__sigma0.25 | 0.2500 | primary | 0.7360 | 0.1760 | 0.1340 | 0.0820 | 0.3180 | 0.5320 | 0.7488 | 8 |
| openai-clip-rn50-quickgelu__cifar10__sigma0.5 | 0.5000 | primary | 0.7360 | 0.1220 | 0.0580 | 0.0960 | 0.5580 | 0.5060 | 0.7952 | 4 |
| openai-clip-rn50-quickgelu__eurosat__sigma0.25 | 0.2500 | primary | 0.3733 | 0.1000 | 0.0733 | 0.0933 | 0.4733 | 0.6533 | 0.8120 | 4 |
| openai-clip-rn50-quickgelu__eurosat__sigma0.5 | 0.5000 | primary | 0.3733 | 0.1600 | 0.0267 | 0.1067 | 0.4200 | 0.5000 | 0.7853 | 4 |
| openai-clip-vit-b16-quickgelu__cifar100__sigma0.25 | 0.2500 | primary | 0.6460 | 0.1860 | 0.1720 | 0.0920 | 0.3880 | 0.5440 | 0.8654 | 57 |
| openai-clip-vit-b16-quickgelu__cifar100__sigma0.5 | 0.5000 | primary | 0.6460 | 0.0540 | 0.1720 | 0.0220 | 0.3520 | 0.6440 | 0.9664 | 21 |
| openai-clip-vit-b16-quickgelu__cifar10__sigma0.25 | 0.2500 | primary | 0.9120 | 0.4720 | 0.2360 | 0.2400 | 0.0920 | 0.2440 | 0.4156 | 10 |
| openai-clip-vit-b16-quickgelu__cifar10__sigma0.5 | 0.5000 | primary | 0.9120 | 0.2620 | 0.3500 | 0.0620 | 0.0840 | 0.3500 | 0.6216 | 9 |
| openai-clip-vit-b16-quickgelu__eurosat__sigma0.25 | 0.2500 | primary | 0.5667 | 0.1267 | 0.1533 | 0.0467 | 0.4533 | 0.8067 | 0.8147 | 7 |
| openai-clip-vit-b16-quickgelu__eurosat__sigma0.5 | 0.5000 | primary | 0.5667 | 0.1467 | 0.2933 | 0.0533 | 0.2200 | 0.4333 | 0.7427 | 4 |
| openclip-vit-b32-laion2b__cifar100__sigma0.25 | 0.2500 | primary | 0.7540 | 0.1320 | 0.1540 | 0.0680 | 0.3720 | 0.5700 | 0.9042 | 47 |
| openclip-vit-b32-laion2b__cifar100__sigma0.5 | 0.5000 | primary | 0.7540 | 0.0500 | 0.1360 | 0.0180 | 0.5000 | 0.7860 | 0.9674 | 25 |
| openclip-vit-b32-laion2b__cifar10__sigma0.25 | 0.2500 | primary | 0.9360 | 0.4160 | 0.1720 | 0.2280 | 0.1400 | 0.2980 | 0.5052 | 10 |
| openclip-vit-b32-laion2b__cifar10__sigma0.5 | 0.5000 | primary | 0.9360 | 0.1540 | 0.1580 | 0.0700 | 0.2620 | 0.7080 | 0.7988 | 8 |
| openclip-vit-b32-laion2b__eurosat__sigma0.25 | 0.2500 | primary | 0.5000 | 0.1267 | 0.1200 | 0.0533 | 0.4333 | 0.7067 | 0.8027 | 5 |
| openclip-vit-b32-laion2b__eurosat__sigma0.5 | 0.5000 | primary | 0.5000 | 0.1000 | 0.0933 | 0.0800 | 0.2933 | 0.9000 | 0.8747 | 4 |

### Sensitivities (descriptive, unapproved sensitivity)

- item-pooled contrast: plain mean of the paired difference over all 6900 primary item positions (CIFAR cells weigh more than EuroSAT); carries no band.
- Rao-Wu finite-stratum band: numpy.random.Generator(numpy.random.PCG64DXSM(seed)), seed 2026091605, 100000 replicates; critical_rao_wu = 0.016037; unscaled PCG64DXSM critical = 0.015481 (Monte-Carlo re-realization of the registered statistic; registered PCG64 critical = 0.015630); ratio = 1.0359; global bound factor sqrt(5/4) = 1.118034

| bank | equal-cell delta | item-pooled delta | pooled - equal-cell | Rao-Wu lower | Rao-Wu upper | RW contains 0 | RW excludes 0 (+) |
|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0000 | 0.0000 | 0.0000 | -0.0160 | 0.0160 | yes | no |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0000 | 0.0000 | 0.0000 | -0.0160 | 0.0160 | yes | no |
| chowers_exact_projected_gap__coefficient_1 | 0.0000 | 0.0000 | 0.0000 | -0.0160 | 0.0160 | yes | no |
| clean_boundary_active__step_0.0025 | -0.0002 | -0.0003 | -0.0001 | -0.0163 | 0.0158 | yes | no |
| clean_boundary_active__step_0.005 | -0.0002 | -0.0006 | -0.0004 | -0.0162 | 0.0159 | yes | no |
| clean_boundary_active__step_0.01 | -0.0001 | -0.0009 | -0.0007 | -0.0162 | 0.0159 | yes | no |
| clean_boundary_active__step_0.02 | -0.0015 | -0.0016 | -0.0001 | -0.0175 | 0.0146 | yes | no |
| clean_boundary_active__step_0.04 | -0.0013 | 0.0000 | 0.0013 | -0.0173 | 0.0147 | yes | no |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0004 | 0.0001 | -0.0002 | -0.0157 | 0.0164 | yes | no |
| cohen_aligned_noisy_margin__step_0.005 | 0.0003 | 0.0000 | -0.0003 | -0.0158 | 0.0163 | yes | no |
| cohen_aligned_noisy_margin__step_0.01 | -0.0001 | -0.0004 | -0.0004 | -0.0161 | 0.0160 | yes | no |
| cohen_aligned_noisy_margin__step_0.02 | 0.0005 | 0.0000 | -0.0005 | -0.0155 | 0.0166 | yes | no |
| cohen_aligned_noisy_margin__step_0.04 | 0.0001 | 0.0001 | 0.0000 | -0.0159 | 0.0161 | yes | no |
| control__learned_shared_translation_pure | -0.0008 | -0.0058 | -0.0050 | -0.0169 | 0.0152 | yes | no |
| control__learned_shared_translation_tangent | 0.0096 | 0.0061 | -0.0035 | -0.0064 | 0.0256 | yes | no |
| control__lowrank_tangent_r8 | 0.1367 | 0.0955 | -0.0412 | 0.1207 | 0.1528 | no | yes |
| control__noisy_class_mean | 0.0734 | 0.0464 | -0.0270 | 0.0574 | 0.0894 | no | yes |
| global_mean_centering__coefficient_0.25 | 0.0026 | 0.0013 | -0.0013 | -0.0135 | 0.0186 | yes | no |
| global_mean_centering__coefficient_0.5 | 0.0075 | 0.0041 | -0.0035 | -0.0085 | 0.0236 | yes | no |
| global_mean_centering__coefficient_1 | 0.0061 | 0.0042 | -0.0019 | -0.0100 | 0.0221 | yes | no |
| gr_clip_style_two_sided__coefficient_1 | 0.0042 | 0.0038 | -0.0004 | -0.0119 | 0.0202 | yes | no |

### Macro over primary cells (every metric, every bank)

| bank | CA@sigma | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | sel=raw | mean radius | top share | Gini | entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0795 | 0.1401 | 0.1147 | 0.0904 | 0.0553 | 0.0593 | 0.3570 | 0.1531 | 0.1573 | 0.6458 | 0.1593 | 0.3823 | 0.5788 | 0.7970 | 0.4343 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0795 | 0.1401 | 0.1147 | 0.0904 | 0.0553 | 0.0593 | 0.3571 | 0.1531 | 0.1573 | 0.6458 | 0.1593 | 0.3823 | 0.5788 | 0.7970 | 0.4343 |
| chowers_exact_projected_gap__coefficient_1 | 0.0795 | 0.1401 | 0.1147 | 0.0904 | 0.0553 | 0.0593 | 0.3571 | 0.1531 | 0.1573 | 0.6458 | 0.1593 | 0.3823 | 0.5788 | 0.7970 | 0.4343 |
| clean_boundary_active__step_0.0025 | 0.0793 | 0.1403 | 0.1161 | 0.0909 | 0.0545 | 0.0595 | 0.3548 | 0.1534 | 0.1579 | 0.6464 | 0.1599 | 0.3811 | 0.5769 | 0.7970 | 0.4362 |
| clean_boundary_active__step_0.005 | 0.0793 | 0.1399 | 0.1171 | 0.0907 | 0.0550 | 0.0597 | 0.3514 | 0.1548 | 0.1580 | 0.6467 | 0.1610 | 0.3801 | 0.5770 | 0.7971 | 0.4367 |
| clean_boundary_active__step_0.01 | 0.0794 | 0.1410 | 0.1164 | 0.0916 | 0.0540 | 0.0599 | 0.3474 | 0.1530 | 0.1586 | 0.6471 | 0.1609 | 0.3784 | 0.5742 | 0.7960 | 0.4416 |
| clean_boundary_active__step_0.02 | 0.0780 | 0.1436 | 0.1168 | 0.0901 | 0.0531 | 0.0605 | 0.3468 | 0.1513 | 0.1614 | 0.6470 | 0.1629 | 0.3766 | 0.5717 | 0.7943 | 0.4443 |
| clean_boundary_active__step_0.04 | 0.0782 | 0.1461 | 0.1168 | 0.0935 | 0.0529 | 0.0624 | 0.3481 | 0.1571 | 0.1641 | 0.6443 | 0.1711 | 0.3797 | 0.5541 | 0.7866 | 0.4555 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0799 | 0.1401 | 0.1158 | 0.0909 | 0.0555 | 0.0597 | 0.3552 | 0.1534 | 0.1577 | 0.6459 | 0.1596 | 0.3819 | 0.5783 | 0.7971 | 0.4351 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0798 | 0.1404 | 0.1159 | 0.0913 | 0.0555 | 0.0601 | 0.3530 | 0.1537 | 0.1583 | 0.6457 | 0.1601 | 0.3817 | 0.5801 | 0.7981 | 0.4334 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0794 | 0.1408 | 0.1165 | 0.0919 | 0.0548 | 0.0603 | 0.3532 | 0.1548 | 0.1575 | 0.6462 | 0.1592 | 0.3812 | 0.5806 | 0.7980 | 0.4353 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0800 | 0.1423 | 0.1166 | 0.0923 | 0.0546 | 0.0604 | 0.3491 | 0.1501 | 0.1619 | 0.6394 | 0.1625 | 0.3809 | 0.5850 | 0.7987 | 0.4342 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0796 | 0.1452 | 0.1177 | 0.0931 | 0.0549 | 0.0612 | 0.3453 | 0.1492 | 0.1642 | 0.6333 | 0.1673 | 0.3816 | 0.5796 | 0.7953 | 0.4432 |
| control__learned_shared_translation_pure | 0.0787 | 0.1293 | 0.1104 | 0.0900 | 0.0546 | 0.0595 | 0.4396 | 0.1089 | 0.1416 | 0.5479 | 0.2396 | 0.4439 | 0.6305 | 0.8386 | 0.3488 |
| control__learned_shared_translation_tangent | 0.0891 | 0.1764 | 0.1385 | 0.1051 | 0.0582 | 0.0745 | 0.2680 | 0.2151 | 0.2081 | 0.6273 | 0.1985 | 0.3208 | 0.4817 | 0.7154 | 0.5641 |
| control__lowrank_tangent_r8 | 0.2163 | 0.3554 | 0.3057 | 0.2489 | 0.1497 | 0.2010 | 0.1231 | 0.2720 | 0.4013 | 0.7746 | 0.4024 | 0.2837 | 0.1277 | 0.3161 | 0.9347 |
| control__noisy_class_mean | 0.1529 | 0.2900 | 0.2355 | 0.1821 | 0.1019 | 0.0720 | 0.1560 | 0.2824 | 0.3400 | 0.2229 | 0.2001 | 0.2704 | 0.1523 | 0.3579 | 0.9161 |
| global_mean_centering__coefficient_0.25 | 0.0821 | 0.1390 | 0.1184 | 0.0918 | 0.0564 | 0.0592 | 0.3777 | 0.1356 | 0.1554 | 0.6443 | 0.1552 | 0.4037 | 0.6340 | 0.8191 | 0.3878 |
| global_mean_centering__coefficient_0.5 | 0.0870 | 0.1362 | 0.1151 | 0.0947 | 0.0638 | 0.0619 | 0.4577 | 0.1115 | 0.1475 | 0.6256 | 0.1620 | 0.4910 | 0.6692 | 0.8392 | 0.3345 |
| global_mean_centering__coefficient_1 | 0.0856 | 0.1249 | 0.1083 | 0.0920 | 0.0650 | 0.0623 | 0.4734 | 0.1272 | 0.1350 | 0.5690 | 0.1791 | 0.5173 | 0.6705 | 0.8529 | 0.3237 |
| gr_clip_style_two_sided__coefficient_1 | 0.0837 | 0.1427 | 0.1214 | 0.0961 | 0.0559 | 0.0711 | 0.3636 | 0.1754 | 0.1621 | 0.6834 | 0.1752 | 0.4311 | 0.5601 | 0.7825 | 0.4512 |
| no_correction | 0.0795 | 0.1401 | 0.1147 | 0.0904 | 0.0553 | 0.0593 | 0.3571 | 0.1531 | 0.1573 | 0.6458 | 0.1593 | 0.3823 | 0.5788 | 0.7970 | 0.4343 |

### Every cell, every bank

#### openai-clip-rn50-quickgelu__cifar100__sigma0.25 (sigma 0.25, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0300 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4180 | 0.2120 | 0.0380 | 0.3880 | 0.6180 | 0.9496 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0300 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4180 | 0.2120 | 0.0380 | 0.3880 | 0.6180 | 0.9496 |
| chowers_exact_projected_gap__coefficient_1 | 0.0300 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4180 | 0.2120 | 0.0380 | 0.3880 | 0.6180 | 0.9496 |
| clean_boundary_active__step_0.0025 | 0.0300 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4160 | 0.2120 | 0.0380 | 0.3940 | 0.6200 | 0.9490 |
| clean_boundary_active__step_0.005 | 0.0300 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4160 | 0.2140 | 0.0380 | 0.3940 | 0.6180 | 0.9505 |
| clean_boundary_active__step_0.01 | 0.0300 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4100 | 0.2140 | 0.0380 | 0.3940 | 0.6200 | 0.9500 |
| clean_boundary_active__step_0.02 | 0.0300 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4080 | 0.2200 | 0.0400 | 0.3940 | 0.6160 | 0.9492 |
| clean_boundary_active__step_0.04 | 0.0320 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4060 | 0.2200 | 0.0400 | 0.3920 | 0.6160 | 0.9486 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0280 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4160 | 0.2160 | 0.0380 | 0.3900 | 0.6180 | 0.9496 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0280 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4160 | 0.2140 | 0.0380 | 0.3900 | 0.6180 | 0.9496 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0280 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4140 | 0.2140 | 0.0400 | 0.3920 | 0.6180 | 0.9488 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0280 | 0.0160 | 0.0100 | 0.0040 | 0.0080 | 0.4080 | 0.2200 | 0.0400 | 0.3820 | 0.6120 | 0.9482 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0280 | 0.0180 | 0.0100 | 0.0040 | 0.0080 | 0.4000 | 0.2280 | 0.0420 | 0.3760 | 0.6040 | 0.9467 |
| control__learned_shared_translation_pure | 0.0120 | 0.0100 | 0.0080 | 0.0080 | 0.0080 | 0.5740 | 0.0820 | 0.0140 | 0.2260 | 0.6840 | 0.9738 |
| control__learned_shared_translation_tangent | 0.0320 | 0.0160 | 0.0100 | 0.0060 | 0.0060 | 0.4260 | 0.2180 | 0.0420 | 0.3940 | 0.6320 | 0.9503 |
| control__lowrank_tangent_r8 | 0.1060 | 0.0760 | 0.0400 | 0.0120 | 0.0240 | 0.1300 | 0.4680 | 0.1440 | 0.3680 | 0.0520 | 0.4964 |
| control__noisy_class_mean | 0.0740 | 0.0560 | 0.0380 | 0.0180 | 0.0120 | 0.2080 | 0.3580 | 0.0860 | 0.0600 | 0.0880 | 0.6430 |
| global_mean_centering__coefficient_0.25 | 0.0280 | 0.0120 | 0.0060 | 0.0040 | 0.0040 | 0.4320 | 0.1880 | 0.0460 | 0.3940 | 0.5960 | 0.9541 |
| global_mean_centering__coefficient_0.5 | 0.0240 | 0.0100 | 0.0060 | 0.0040 | 0.0040 | 0.3380 | 0.2040 | 0.0360 | 0.3700 | 0.4740 | 0.9561 |
| global_mean_centering__coefficient_1 | 0.0260 | 0.0220 | 0.0160 | 0.0060 | 0.0100 | 0.3420 | 0.2420 | 0.0320 | 0.2760 | 0.3860 | 0.9422 |
| gr_clip_style_two_sided__coefficient_1 | 0.0320 | 0.0280 | 0.0080 | 0.0060 | 0.0060 | 0.3200 | 0.2420 | 0.0420 | 0.4260 | 0.4160 | 0.9359 |
| no_correction | 0.0300 | 0.0140 | 0.0100 | 0.0040 | 0.0080 | 0.4180 | 0.2120 | 0.0380 | 0.3880 | 0.6180 | 0.9496 |

#### openai-clip-rn50-quickgelu__cifar100__sigma0.5 (sigma 0.5, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6500 | 0.0420 | 0.0100 | 0.3880 | 0.6180 | 0.9764 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6500 | 0.0420 | 0.0100 | 0.3880 | 0.6180 | 0.9764 |
| chowers_exact_projected_gap__coefficient_1 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6500 | 0.0420 | 0.0100 | 0.3880 | 0.6180 | 0.9764 |
| clean_boundary_active__step_0.0025 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6480 | 0.0440 | 0.0120 | 0.3940 | 0.6140 | 0.9761 |
| clean_boundary_active__step_0.005 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6460 | 0.0420 | 0.0120 | 0.3940 | 0.6140 | 0.9761 |
| clean_boundary_active__step_0.01 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6460 | 0.0480 | 0.0120 | 0.3940 | 0.6120 | 0.9760 |
| clean_boundary_active__step_0.02 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6420 | 0.0560 | 0.0120 | 0.3940 | 0.6120 | 0.9760 |
| clean_boundary_active__step_0.04 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6360 | 0.0600 | 0.0120 | 0.3920 | 0.6040 | 0.9762 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6500 | 0.0440 | 0.0100 | 0.3880 | 0.6160 | 0.9762 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6500 | 0.0440 | 0.0100 | 0.3900 | 0.6160 | 0.9762 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6480 | 0.0460 | 0.0100 | 0.3900 | 0.6160 | 0.9762 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6480 | 0.0440 | 0.0100 | 0.3820 | 0.6180 | 0.9760 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6500 | 0.0500 | 0.0100 | 0.3760 | 0.6180 | 0.9758 |
| control__learned_shared_translation_pure | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6680 | 0.0440 | 0.0100 | 0.2960 | 0.5840 | 0.9795 |
| control__learned_shared_translation_tangent | 0.0100 | 0.0080 | 0.0060 | 0.0040 | 0.0000 | 0.3440 | 0.2220 | 0.0120 | 0.4020 | 0.2680 | 0.9401 |
| control__lowrank_tangent_r8 | 0.0460 | 0.0360 | 0.0320 | 0.0160 | 0.0060 | 0.1660 | 0.4060 | 0.0680 | 0.3920 | 0.0720 | 0.5963 |
| control__noisy_class_mean | 0.0420 | 0.0360 | 0.0340 | 0.0300 | 0.0000 | 0.2700 | 0.2900 | 0.0440 | 0.0160 | 0.1360 | 0.7988 |
| global_mean_centering__coefficient_0.25 | 0.0120 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6720 | 0.0580 | 0.0140 | 0.3940 | 0.6520 | 0.9772 |
| global_mean_centering__coefficient_0.5 | 0.0140 | 0.0120 | 0.0060 | 0.0040 | 0.0020 | 0.6500 | 0.0720 | 0.0140 | 0.3700 | 0.6440 | 0.9747 |
| global_mean_centering__coefficient_1 | 0.0080 | 0.0080 | 0.0060 | 0.0040 | 0.0000 | 0.4640 | 0.1340 | 0.0080 | 0.2760 | 0.5220 | 0.9646 |
| gr_clip_style_two_sided__coefficient_1 | 0.0140 | 0.0140 | 0.0120 | 0.0120 | 0.0060 | 0.5720 | 0.0740 | 0.0140 | 0.4260 | 0.5220 | 0.9712 |
| no_correction | 0.0100 | 0.0100 | 0.0100 | 0.0100 | 0.0020 | 0.6500 | 0.0420 | 0.0100 | 0.3880 | 0.6180 | 0.9764 |

#### openai-clip-rn50-quickgelu__cifar10__sigma0.25 (sigma 0.25, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1640 | 0.1300 | 0.0820 | 0.0320 | 0.0640 | 0.3180 | 0.1340 | 0.1760 | 0.7360 | 0.5320 | 0.7488 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1640 | 0.1300 | 0.0820 | 0.0320 | 0.0640 | 0.3180 | 0.1340 | 0.1760 | 0.7360 | 0.5320 | 0.7488 |
| chowers_exact_projected_gap__coefficient_1 | 0.1640 | 0.1300 | 0.0820 | 0.0320 | 0.0640 | 0.3180 | 0.1340 | 0.1760 | 0.7360 | 0.5320 | 0.7488 |
| clean_boundary_active__step_0.0025 | 0.1640 | 0.1280 | 0.0820 | 0.0320 | 0.0640 | 0.3160 | 0.1260 | 0.1760 | 0.7360 | 0.5280 | 0.7492 |
| clean_boundary_active__step_0.005 | 0.1660 | 0.1280 | 0.0760 | 0.0320 | 0.0620 | 0.3120 | 0.1200 | 0.1780 | 0.7360 | 0.5220 | 0.7480 |
| clean_boundary_active__step_0.01 | 0.1680 | 0.1300 | 0.0780 | 0.0320 | 0.0640 | 0.3100 | 0.1140 | 0.1800 | 0.7360 | 0.5120 | 0.7444 |
| clean_boundary_active__step_0.02 | 0.1680 | 0.1260 | 0.0800 | 0.0320 | 0.0680 | 0.3080 | 0.1120 | 0.1780 | 0.7380 | 0.5020 | 0.7448 |
| clean_boundary_active__step_0.04 | 0.1680 | 0.1260 | 0.0880 | 0.0340 | 0.0760 | 0.3120 | 0.1020 | 0.1740 | 0.7300 | 0.4900 | 0.7484 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1660 | 0.1280 | 0.0820 | 0.0320 | 0.0640 | 0.3180 | 0.1260 | 0.1760 | 0.7360 | 0.5280 | 0.7480 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1680 | 0.1280 | 0.0800 | 0.0320 | 0.0640 | 0.3140 | 0.1180 | 0.1780 | 0.7360 | 0.5240 | 0.7480 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1680 | 0.1280 | 0.0840 | 0.0320 | 0.0680 | 0.3100 | 0.1160 | 0.1800 | 0.7360 | 0.5120 | 0.7440 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1700 | 0.1280 | 0.0860 | 0.0340 | 0.0700 | 0.3140 | 0.1180 | 0.1800 | 0.7340 | 0.5140 | 0.7432 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1760 | 0.1320 | 0.0860 | 0.0320 | 0.0700 | 0.3020 | 0.1300 | 0.1820 | 0.7280 | 0.4860 | 0.7344 |
| control__learned_shared_translation_pure | 0.1640 | 0.1300 | 0.0800 | 0.0340 | 0.0620 | 0.3160 | 0.1300 | 0.1760 | 0.7280 | 0.5380 | 0.7572 |
| control__learned_shared_translation_tangent | 0.2160 | 0.1540 | 0.0860 | 0.0300 | 0.0720 | 0.1880 | 0.2180 | 0.2420 | 0.7400 | 0.3900 | 0.5916 |
| control__lowrank_tangent_r8 | 0.3520 | 0.2900 | 0.2160 | 0.0980 | 0.1920 | 0.2220 | 0.1640 | 0.3920 | 0.7620 | 0.1520 | 0.1024 |
| control__noisy_class_mean | 0.2300 | 0.1780 | 0.1220 | 0.0500 | 0.0580 | 0.3320 | 0.1300 | 0.2580 | 0.2920 | 0.1440 | 0.2124 |
| global_mean_centering__coefficient_0.25 | 0.1460 | 0.1140 | 0.0880 | 0.0340 | 0.0680 | 0.4120 | 0.0800 | 0.1600 | 0.7200 | 0.6940 | 0.8036 |
| global_mean_centering__coefficient_0.5 | 0.1280 | 0.1100 | 0.0980 | 0.0680 | 0.0800 | 0.6220 | 0.0300 | 0.1300 | 0.6860 | 0.8420 | 0.8512 |
| global_mean_centering__coefficient_1 | 0.1100 | 0.1060 | 0.1000 | 0.0920 | 0.0880 | 0.7600 | 0.0200 | 0.1120 | 0.6320 | 0.9380 | 0.8832 |
| gr_clip_style_two_sided__coefficient_1 | 0.1480 | 0.1160 | 0.0800 | 0.0320 | 0.0660 | 0.2900 | 0.1600 | 0.1640 | 0.7780 | 0.6320 | 0.7256 |
| no_correction | 0.1640 | 0.1300 | 0.0820 | 0.0320 | 0.0640 | 0.3180 | 0.1340 | 0.1760 | 0.7360 | 0.5320 | 0.7488 |

#### openai-clip-rn50-quickgelu__cifar10__sigma0.5 (sigma 0.5, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1160 | 0.1100 | 0.1040 | 0.0960 | 0.0860 | 0.5580 | 0.0580 | 0.1220 | 0.7360 | 0.5060 | 0.7952 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1160 | 0.1100 | 0.1040 | 0.0960 | 0.0860 | 0.5580 | 0.0580 | 0.1220 | 0.7360 | 0.5060 | 0.7952 |
| chowers_exact_projected_gap__coefficient_1 | 0.1160 | 0.1100 | 0.1040 | 0.0960 | 0.0860 | 0.5580 | 0.0580 | 0.1220 | 0.7360 | 0.5060 | 0.7952 |
| clean_boundary_active__step_0.0025 | 0.1180 | 0.1120 | 0.1060 | 0.0980 | 0.0880 | 0.5680 | 0.0480 | 0.1220 | 0.7360 | 0.4920 | 0.7932 |
| clean_boundary_active__step_0.005 | 0.1220 | 0.1120 | 0.1060 | 0.0980 | 0.0880 | 0.5640 | 0.0360 | 0.1240 | 0.7360 | 0.4980 | 0.7948 |
| clean_boundary_active__step_0.01 | 0.1200 | 0.1100 | 0.1080 | 0.0940 | 0.0860 | 0.5720 | 0.0400 | 0.1240 | 0.7360 | 0.5060 | 0.7964 |
| clean_boundary_active__step_0.02 | 0.1200 | 0.1160 | 0.1100 | 0.0920 | 0.0840 | 0.5860 | 0.0280 | 0.1240 | 0.7380 | 0.5300 | 0.8020 |
| clean_boundary_active__step_0.04 | 0.1180 | 0.1140 | 0.1100 | 0.1000 | 0.0920 | 0.6320 | 0.0480 | 0.1180 | 0.7300 | 0.5880 | 0.8156 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1180 | 0.1120 | 0.1060 | 0.0980 | 0.0880 | 0.5660 | 0.0480 | 0.1220 | 0.7360 | 0.4920 | 0.7932 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1220 | 0.1120 | 0.1060 | 0.0980 | 0.0880 | 0.5640 | 0.0360 | 0.1240 | 0.7360 | 0.4980 | 0.7948 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1200 | 0.1100 | 0.1080 | 0.0940 | 0.0860 | 0.5740 | 0.0360 | 0.1240 | 0.7360 | 0.5060 | 0.7964 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1200 | 0.1160 | 0.1100 | 0.0920 | 0.0820 | 0.5800 | 0.0380 | 0.1240 | 0.7300 | 0.5280 | 0.8016 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1180 | 0.1140 | 0.1100 | 0.1000 | 0.0880 | 0.6220 | 0.0540 | 0.1180 | 0.7180 | 0.5780 | 0.8136 |
| control__learned_shared_translation_pure | 0.1180 | 0.1040 | 0.1020 | 0.0880 | 0.0800 | 0.5540 | 0.0420 | 0.1220 | 0.7280 | 0.4940 | 0.7900 |
| control__learned_shared_translation_tangent | 0.1360 | 0.1200 | 0.1080 | 0.0860 | 0.0600 | 0.4600 | 0.1040 | 0.1520 | 0.7600 | 0.6440 | 0.7644 |
| control__lowrank_tangent_r8 | 0.2280 | 0.1940 | 0.1580 | 0.1220 | 0.1060 | 0.2820 | 0.1460 | 0.2400 | 0.7620 | 0.1480 | 0.1660 |
| control__noisy_class_mean | 0.1540 | 0.1280 | 0.1060 | 0.0780 | 0.0220 | 0.3880 | 0.1060 | 0.1700 | 0.1060 | 0.2540 | 0.3900 |
| global_mean_centering__coefficient_0.25 | 0.1280 | 0.1160 | 0.1060 | 0.0960 | 0.0820 | 0.5740 | 0.0440 | 0.1280 | 0.7200 | 0.6240 | 0.8236 |
| global_mean_centering__coefficient_0.5 | 0.1260 | 0.1260 | 0.1160 | 0.1040 | 0.0900 | 0.6900 | 0.0080 | 0.1260 | 0.6860 | 0.8280 | 0.8640 |
| global_mean_centering__coefficient_1 | 0.1120 | 0.1060 | 0.1040 | 0.0960 | 0.0840 | 0.8160 | 0.0100 | 0.1160 | 0.6320 | 0.9420 | 0.8880 |
| gr_clip_style_two_sided__coefficient_1 | 0.1260 | 0.1140 | 0.1000 | 0.0820 | 0.0640 | 0.4780 | 0.0980 | 0.1320 | 0.7780 | 0.5460 | 0.7736 |
| no_correction | 0.1160 | 0.1100 | 0.1040 | 0.0960 | 0.0860 | 0.5580 | 0.0580 | 0.1220 | 0.7360 | 0.5060 | 0.7952 |

#### openai-clip-rn50-quickgelu__eurosat__sigma0.25 (sigma 0.25, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4733 | 0.0733 | 0.1000 | 0.3733 | 0.6533 | 0.8120 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4733 | 0.0733 | 0.1000 | 0.3733 | 0.6533 | 0.8120 |
| chowers_exact_projected_gap__coefficient_1 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4733 | 0.0733 | 0.1000 | 0.3733 | 0.6533 | 0.8120 |
| clean_boundary_active__step_0.0025 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4733 | 0.0667 | 0.1000 | 0.3733 | 0.6667 | 0.8120 |
| clean_boundary_active__step_0.005 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4667 | 0.0667 | 0.1000 | 0.3733 | 0.6667 | 0.8120 |
| clean_boundary_active__step_0.01 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4667 | 0.0533 | 0.1000 | 0.3733 | 0.6733 | 0.8133 |
| clean_boundary_active__step_0.02 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4733 | 0.0667 | 0.1000 | 0.3733 | 0.6733 | 0.8133 |
| clean_boundary_active__step_0.04 | 0.1000 | 0.0933 | 0.0933 | 0.0867 | 0.0267 | 0.4800 | 0.0467 | 0.1067 | 0.3600 | 0.7000 | 0.8147 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4600 | 0.0867 | 0.1000 | 0.3733 | 0.6533 | 0.8093 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4400 | 0.1067 | 0.1000 | 0.3733 | 0.6533 | 0.8093 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4267 | 0.1133 | 0.1000 | 0.3800 | 0.6467 | 0.8080 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4067 | 0.1200 | 0.1267 | 0.3600 | 0.5933 | 0.7933 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1533 | 0.1000 | 0.0867 | 0.0800 | 0.0267 | 0.3400 | 0.0733 | 0.1600 | 0.3467 | 0.5333 | 0.7760 |
| control__learned_shared_translation_pure | 0.1933 | 0.1533 | 0.1200 | 0.0733 | 0.0467 | 0.4933 | 0.0200 | 0.1933 | 0.2133 | 0.5733 | 0.8147 |
| control__learned_shared_translation_tangent | 0.2933 | 0.2533 | 0.2267 | 0.1667 | 0.1067 | 0.3800 | 0.0867 | 0.3067 | 0.3533 | 0.4933 | 0.7320 |
| control__lowrank_tangent_r8 | 0.5267 | 0.4867 | 0.4267 | 0.2867 | 0.4133 | 0.1600 | 0.0800 | 0.5467 | 0.8267 | 0.1600 | 0.2440 |
| control__noisy_class_mean | 0.4933 | 0.4267 | 0.3533 | 0.2533 | 0.1000 | 0.2400 | 0.0733 | 0.5133 | 0.1000 | 0.2200 | 0.3120 |
| global_mean_centering__coefficient_0.25 | 0.1400 | 0.1200 | 0.1067 | 0.0800 | 0.0267 | 0.4267 | 0.1267 | 0.1667 | 0.3600 | 0.5200 | 0.7853 |
| global_mean_centering__coefficient_0.5 | 0.2133 | 0.1533 | 0.1133 | 0.0800 | 0.0067 | 0.4533 | 0.0733 | 0.2133 | 0.3200 | 0.4600 | 0.7440 |
| global_mean_centering__coefficient_1 | 0.1000 | 0.0933 | 0.0800 | 0.0600 | 0.0000 | 0.6533 | 0.0467 | 0.1000 | 0.2067 | 0.8333 | 0.8627 |
| gr_clip_style_two_sided__coefficient_1 | 0.1333 | 0.1200 | 0.1200 | 0.0733 | 0.0533 | 0.5133 | 0.0533 | 0.1467 | 0.4533 | 0.6467 | 0.7693 |
| no_correction | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0267 | 0.4733 | 0.0733 | 0.1000 | 0.3733 | 0.6533 | 0.8120 |

#### openai-clip-rn50-quickgelu__eurosat__sigma0.5 (sigma 0.5, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1600 | 0.1467 | 0.1400 | 0.1067 | 0.0333 | 0.4200 | 0.0267 | 0.1600 | 0.3733 | 0.5000 | 0.7853 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1600 | 0.1467 | 0.1400 | 0.1067 | 0.0333 | 0.4200 | 0.0267 | 0.1600 | 0.3733 | 0.5000 | 0.7853 |
| chowers_exact_projected_gap__coefficient_1 | 0.1600 | 0.1467 | 0.1400 | 0.1067 | 0.0333 | 0.4200 | 0.0267 | 0.1600 | 0.3733 | 0.5000 | 0.7853 |
| clean_boundary_active__step_0.0025 | 0.1600 | 0.1533 | 0.1400 | 0.1067 | 0.0333 | 0.4133 | 0.0267 | 0.1600 | 0.3733 | 0.5000 | 0.7853 |
| clean_boundary_active__step_0.005 | 0.1600 | 0.1533 | 0.1400 | 0.1067 | 0.0333 | 0.4133 | 0.0267 | 0.1600 | 0.3733 | 0.5000 | 0.7853 |
| clean_boundary_active__step_0.01 | 0.1600 | 0.1533 | 0.1400 | 0.1067 | 0.0333 | 0.4133 | 0.0267 | 0.1600 | 0.3733 | 0.5067 | 0.7867 |
| clean_boundary_active__step_0.02 | 0.1667 | 0.1533 | 0.1400 | 0.1000 | 0.0333 | 0.4000 | 0.0200 | 0.1733 | 0.3733 | 0.5267 | 0.7907 |
| clean_boundary_active__step_0.04 | 0.1667 | 0.1600 | 0.1400 | 0.1000 | 0.0400 | 0.3733 | 0.0267 | 0.1733 | 0.3600 | 0.5267 | 0.7907 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1600 | 0.1533 | 0.1400 | 0.1067 | 0.0333 | 0.4133 | 0.0267 | 0.1600 | 0.3733 | 0.5000 | 0.7853 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1600 | 0.1467 | 0.1400 | 0.1067 | 0.0333 | 0.4067 | 0.0200 | 0.1600 | 0.3733 | 0.5067 | 0.7867 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1600 | 0.1533 | 0.1400 | 0.1000 | 0.0333 | 0.4067 | 0.0267 | 0.1667 | 0.3800 | 0.5200 | 0.7893 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1667 | 0.1533 | 0.1333 | 0.0933 | 0.0333 | 0.3800 | 0.0333 | 0.1733 | 0.3600 | 0.5267 | 0.7907 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1600 | 0.1533 | 0.1333 | 0.0933 | 0.0400 | 0.3667 | 0.0600 | 0.1667 | 0.3467 | 0.5200 | 0.7800 |
| control__learned_shared_translation_pure | 0.1267 | 0.1200 | 0.1133 | 0.0667 | 0.0000 | 0.5200 | 0.0333 | 0.1267 | 0.2533 | 0.6400 | 0.8093 |
| control__learned_shared_translation_tangent | 0.1867 | 0.1733 | 0.1533 | 0.1067 | 0.1000 | 0.2533 | 0.1267 | 0.1933 | 0.3600 | 0.5267 | 0.7293 |
| control__lowrank_tangent_r8 | 0.4067 | 0.3800 | 0.3267 | 0.2667 | 0.2467 | 0.2067 | 0.1400 | 0.4533 | 0.7933 | 0.2200 | 0.3213 |
| control__noisy_class_mean | 0.3467 | 0.3067 | 0.2600 | 0.1933 | 0.0867 | 0.2267 | 0.0933 | 0.3733 | 0.1667 | 0.2067 | 0.3827 |
| global_mean_centering__coefficient_0.25 | 0.1467 | 0.1400 | 0.1067 | 0.0800 | 0.0067 | 0.4133 | 0.0333 | 0.1600 | 0.3600 | 0.7267 | 0.8387 |
| global_mean_centering__coefficient_0.5 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0000 | 0.8933 | 0.0000 | 0.1000 | 0.3200 | 0.9933 | 0.8987 |
| global_mean_centering__coefficient_1 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0000 | 0.8933 | 0.0000 | 0.1000 | 0.2067 | 0.9933 | 0.8987 |
| gr_clip_style_two_sided__coefficient_1 | 0.0933 | 0.0933 | 0.0933 | 0.0800 | 0.0000 | 0.4800 | 0.0600 | 0.0933 | 0.4533 | 0.8733 | 0.8733 |
| no_correction | 0.1600 | 0.1467 | 0.1400 | 0.1067 | 0.0333 | 0.4200 | 0.0267 | 0.1600 | 0.3733 | 0.5000 | 0.7853 |

#### openai-clip-vit-b16-quickgelu__cifar100__sigma0.25 (sigma 0.25, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1620 | 0.1200 | 0.0920 | 0.0500 | 0.0860 | 0.3880 | 0.1720 | 0.1860 | 0.6460 | 0.5440 | 0.8654 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1620 | 0.1200 | 0.0920 | 0.0500 | 0.0860 | 0.3880 | 0.1720 | 0.1860 | 0.6460 | 0.5440 | 0.8654 |
| chowers_exact_projected_gap__coefficient_1 | 0.1620 | 0.1200 | 0.0920 | 0.0500 | 0.0860 | 0.3880 | 0.1720 | 0.1860 | 0.6460 | 0.5440 | 0.8654 |
| clean_boundary_active__step_0.0025 | 0.1620 | 0.1220 | 0.0920 | 0.0500 | 0.0860 | 0.3900 | 0.1680 | 0.1860 | 0.6460 | 0.5440 | 0.8634 |
| clean_boundary_active__step_0.005 | 0.1620 | 0.1220 | 0.0920 | 0.0500 | 0.0860 | 0.3900 | 0.1680 | 0.1860 | 0.6480 | 0.5460 | 0.8636 |
| clean_boundary_active__step_0.01 | 0.1600 | 0.1220 | 0.0920 | 0.0500 | 0.0860 | 0.3880 | 0.1740 | 0.1860 | 0.6520 | 0.5440 | 0.8631 |
| clean_boundary_active__step_0.02 | 0.1580 | 0.1220 | 0.0920 | 0.0500 | 0.0860 | 0.3880 | 0.1780 | 0.1840 | 0.6560 | 0.5380 | 0.8616 |
| clean_boundary_active__step_0.04 | 0.1600 | 0.1240 | 0.0940 | 0.0500 | 0.0880 | 0.3860 | 0.1720 | 0.1860 | 0.6640 | 0.5400 | 0.8600 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1620 | 0.1200 | 0.0920 | 0.0500 | 0.0860 | 0.3880 | 0.1740 | 0.1860 | 0.6460 | 0.5440 | 0.8654 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1620 | 0.1200 | 0.0920 | 0.0500 | 0.0860 | 0.3900 | 0.1760 | 0.1860 | 0.6460 | 0.5420 | 0.8653 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1620 | 0.1200 | 0.0920 | 0.0500 | 0.0860 | 0.3900 | 0.1760 | 0.1840 | 0.6420 | 0.5460 | 0.8664 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1620 | 0.1200 | 0.0920 | 0.0500 | 0.0860 | 0.3920 | 0.1720 | 0.1860 | 0.6400 | 0.5460 | 0.8642 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1600 | 0.1200 | 0.0920 | 0.0520 | 0.0840 | 0.3940 | 0.1620 | 0.1860 | 0.6360 | 0.5460 | 0.8646 |
| control__learned_shared_translation_pure | 0.0960 | 0.0860 | 0.0700 | 0.0360 | 0.0640 | 0.4500 | 0.1240 | 0.1080 | 0.4100 | 0.6220 | 0.9450 |
| control__learned_shared_translation_tangent | 0.1540 | 0.1260 | 0.0860 | 0.0480 | 0.0780 | 0.3580 | 0.1600 | 0.1740 | 0.6500 | 0.5300 | 0.8689 |
| control__lowrank_tangent_r8 | 0.2740 | 0.2120 | 0.1640 | 0.0760 | 0.1460 | 0.0660 | 0.4120 | 0.3220 | 0.6920 | 0.0360 | 0.4054 |
| control__noisy_class_mean | 0.2620 | 0.1860 | 0.1280 | 0.0640 | 0.0740 | 0.0720 | 0.4320 | 0.3300 | 0.3280 | 0.0420 | 0.3500 |
| global_mean_centering__coefficient_0.25 | 0.1580 | 0.1300 | 0.0940 | 0.0500 | 0.0860 | 0.3360 | 0.1740 | 0.1780 | 0.6760 | 0.5000 | 0.8637 |
| global_mean_centering__coefficient_0.5 | 0.1620 | 0.1240 | 0.0900 | 0.0400 | 0.0840 | 0.2360 | 0.2060 | 0.1800 | 0.6400 | 0.3700 | 0.8732 |
| global_mean_centering__coefficient_1 | 0.1380 | 0.1020 | 0.0840 | 0.0360 | 0.0740 | 0.1740 | 0.2580 | 0.1560 | 0.5320 | 0.2860 | 0.8812 |
| gr_clip_style_two_sided__coefficient_1 | 0.1860 | 0.1460 | 0.1120 | 0.0440 | 0.0980 | 0.2040 | 0.2520 | 0.2120 | 0.6860 | 0.3540 | 0.8102 |
| no_correction | 0.1620 | 0.1200 | 0.0920 | 0.0500 | 0.0860 | 0.3880 | 0.1720 | 0.1860 | 0.6460 | 0.5440 | 0.8654 |

#### openai-clip-vit-b16-quickgelu__cifar100__sigma0.5 (sigma 0.5, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0480 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3520 | 0.1720 | 0.0540 | 0.6460 | 0.6440 | 0.9664 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0480 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3520 | 0.1720 | 0.0540 | 0.6460 | 0.6440 | 0.9664 |
| chowers_exact_projected_gap__coefficient_1 | 0.0480 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3520 | 0.1720 | 0.0540 | 0.6460 | 0.6440 | 0.9664 |
| clean_boundary_active__step_0.0025 | 0.0480 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3520 | 0.1740 | 0.0520 | 0.6460 | 0.6420 | 0.9656 |
| clean_boundary_active__step_0.005 | 0.0480 | 0.0380 | 0.0320 | 0.0220 | 0.0180 | 0.3500 | 0.1760 | 0.0520 | 0.6480 | 0.6400 | 0.9656 |
| clean_boundary_active__step_0.01 | 0.0460 | 0.0400 | 0.0320 | 0.0200 | 0.0160 | 0.3500 | 0.1840 | 0.0540 | 0.6520 | 0.6380 | 0.9650 |
| clean_boundary_active__step_0.02 | 0.0460 | 0.0400 | 0.0340 | 0.0200 | 0.0160 | 0.3480 | 0.1820 | 0.0540 | 0.6560 | 0.6300 | 0.9648 |
| clean_boundary_active__step_0.04 | 0.0500 | 0.0400 | 0.0360 | 0.0220 | 0.0180 | 0.3300 | 0.1820 | 0.0560 | 0.6640 | 0.6020 | 0.9627 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0480 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3540 | 0.1700 | 0.0540 | 0.6460 | 0.6460 | 0.9660 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0480 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3560 | 0.1740 | 0.0520 | 0.6420 | 0.6460 | 0.9662 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0460 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3600 | 0.1760 | 0.0520 | 0.6400 | 0.6500 | 0.9663 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0460 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3620 | 0.1760 | 0.0520 | 0.6400 | 0.6580 | 0.9662 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0460 | 0.0400 | 0.0300 | 0.0220 | 0.0180 | 0.3700 | 0.1800 | 0.0540 | 0.6320 | 0.6680 | 0.9661 |
| control__learned_shared_translation_pure | 0.0320 | 0.0280 | 0.0240 | 0.0140 | 0.0120 | 0.3700 | 0.1360 | 0.0340 | 0.4060 | 0.6360 | 0.9763 |
| control__learned_shared_translation_tangent | 0.0440 | 0.0380 | 0.0320 | 0.0200 | 0.0180 | 0.3540 | 0.1960 | 0.0600 | 0.6480 | 0.6280 | 0.9636 |
| control__lowrank_tangent_r8 | 0.0980 | 0.0840 | 0.0720 | 0.0360 | 0.0320 | 0.0500 | 0.6460 | 0.1640 | 0.6980 | 0.0640 | 0.6381 |
| control__noisy_class_mean | 0.1080 | 0.0900 | 0.0620 | 0.0280 | 0.0120 | 0.0520 | 0.6640 | 0.1700 | 0.1220 | 0.0740 | 0.5364 |
| global_mean_centering__coefficient_0.25 | 0.0500 | 0.0460 | 0.0380 | 0.0120 | 0.0100 | 0.2860 | 0.2260 | 0.0560 | 0.6760 | 0.5120 | 0.9619 |
| global_mean_centering__coefficient_0.5 | 0.0460 | 0.0360 | 0.0300 | 0.0180 | 0.0180 | 0.1900 | 0.2460 | 0.0560 | 0.6400 | 0.3860 | 0.9554 |
| global_mean_centering__coefficient_1 | 0.0360 | 0.0340 | 0.0240 | 0.0140 | 0.0140 | 0.1960 | 0.2320 | 0.0500 | 0.5320 | 0.3920 | 0.9566 |
| gr_clip_style_two_sided__coefficient_1 | 0.0560 | 0.0440 | 0.0320 | 0.0200 | 0.0180 | 0.1560 | 0.4080 | 0.0760 | 0.6860 | 0.2900 | 0.9160 |
| no_correction | 0.0480 | 0.0380 | 0.0300 | 0.0220 | 0.0180 | 0.3520 | 0.1720 | 0.0540 | 0.6460 | 0.6440 | 0.9664 |

#### openai-clip-vit-b16-quickgelu__cifar10__sigma0.25 (sigma 0.25, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.4380 | 0.3260 | 0.2400 | 0.1040 | 0.2380 | 0.0920 | 0.2360 | 0.4720 | 0.9120 | 0.2440 | 0.4156 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.4380 | 0.3260 | 0.2400 | 0.1040 | 0.2380 | 0.0920 | 0.2360 | 0.4720 | 0.9120 | 0.2440 | 0.4156 |
| chowers_exact_projected_gap__coefficient_1 | 0.4380 | 0.3260 | 0.2400 | 0.1040 | 0.2380 | 0.0920 | 0.2360 | 0.4720 | 0.9120 | 0.2440 | 0.4156 |
| clean_boundary_active__step_0.0025 | 0.4380 | 0.3280 | 0.2400 | 0.1040 | 0.2380 | 0.0920 | 0.2400 | 0.4740 | 0.9120 | 0.2440 | 0.4188 |
| clean_boundary_active__step_0.005 | 0.4380 | 0.3280 | 0.2400 | 0.1060 | 0.2380 | 0.0920 | 0.2360 | 0.4760 | 0.9120 | 0.2440 | 0.4196 |
| clean_boundary_active__step_0.01 | 0.4380 | 0.3280 | 0.2420 | 0.1060 | 0.2400 | 0.0920 | 0.2340 | 0.4740 | 0.9120 | 0.2440 | 0.4256 |
| clean_boundary_active__step_0.02 | 0.4340 | 0.3260 | 0.2400 | 0.1040 | 0.2380 | 0.0900 | 0.2440 | 0.4800 | 0.9120 | 0.2340 | 0.4216 |
| clean_boundary_active__step_0.04 | 0.4300 | 0.3240 | 0.2520 | 0.1080 | 0.2500 | 0.0860 | 0.2400 | 0.4820 | 0.9160 | 0.2280 | 0.4216 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.4380 | 0.3260 | 0.2380 | 0.1040 | 0.2360 | 0.0940 | 0.2360 | 0.4720 | 0.9120 | 0.2440 | 0.4164 |
| cohen_aligned_noisy_margin__step_0.005 | 0.4380 | 0.3260 | 0.2380 | 0.1040 | 0.2360 | 0.0940 | 0.2360 | 0.4720 | 0.9120 | 0.2440 | 0.4168 |
| cohen_aligned_noisy_margin__step_0.01 | 0.4380 | 0.3280 | 0.2380 | 0.1040 | 0.2360 | 0.0940 | 0.2340 | 0.4720 | 0.9120 | 0.2440 | 0.4168 |
| cohen_aligned_noisy_margin__step_0.02 | 0.4380 | 0.3260 | 0.2400 | 0.1060 | 0.2380 | 0.0940 | 0.2320 | 0.4780 | 0.9120 | 0.2420 | 0.4152 |
| cohen_aligned_noisy_margin__step_0.04 | 0.4320 | 0.3260 | 0.2420 | 0.1100 | 0.2400 | 0.0940 | 0.2400 | 0.4720 | 0.9120 | 0.2480 | 0.4252 |
| control__learned_shared_translation_pure | 0.3760 | 0.2960 | 0.2100 | 0.0960 | 0.2080 | 0.1480 | 0.2040 | 0.4380 | 0.9160 | 0.3680 | 0.5292 |
| control__learned_shared_translation_tangent | 0.4560 | 0.3540 | 0.2500 | 0.1120 | 0.2460 | 0.0880 | 0.2060 | 0.5060 | 0.9180 | 0.2960 | 0.3872 |
| control__lowrank_tangent_r8 | 0.5740 | 0.4920 | 0.3780 | 0.1780 | 0.3740 | 0.0960 | 0.1360 | 0.6240 | 0.9460 | 0.1220 | 0.0944 |
| control__noisy_class_mean | 0.4860 | 0.3800 | 0.3060 | 0.1520 | 0.2280 | 0.1480 | 0.1360 | 0.5340 | 0.5700 | 0.1940 | 0.1708 |
| global_mean_centering__coefficient_0.25 | 0.4240 | 0.3360 | 0.2420 | 0.1100 | 0.2400 | 0.0980 | 0.2260 | 0.4460 | 0.9100 | 0.2620 | 0.4500 |
| global_mean_centering__coefficient_0.5 | 0.4040 | 0.3320 | 0.2400 | 0.1160 | 0.2360 | 0.1140 | 0.1860 | 0.4300 | 0.8840 | 0.3400 | 0.5096 |
| global_mean_centering__coefficient_1 | 0.3960 | 0.3140 | 0.2520 | 0.1240 | 0.2480 | 0.1600 | 0.1500 | 0.4140 | 0.8400 | 0.4340 | 0.5640 |
| gr_clip_style_two_sided__coefficient_1 | 0.4500 | 0.3600 | 0.2500 | 0.1080 | 0.2440 | 0.0820 | 0.2260 | 0.5200 | 0.9160 | 0.2140 | 0.3816 |
| no_correction | 0.4380 | 0.3260 | 0.2400 | 0.1040 | 0.2380 | 0.0920 | 0.2360 | 0.4720 | 0.9120 | 0.2440 | 0.4156 |

#### openai-clip-vit-b16-quickgelu__cifar10__sigma0.5 (sigma 0.5, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.2180 | 0.1800 | 0.1200 | 0.0620 | 0.0620 | 0.0840 | 0.3500 | 0.2620 | 0.9120 | 0.3500 | 0.6216 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.2180 | 0.1800 | 0.1200 | 0.0620 | 0.0620 | 0.0840 | 0.3500 | 0.2620 | 0.9120 | 0.3500 | 0.6216 |
| chowers_exact_projected_gap__coefficient_1 | 0.2180 | 0.1800 | 0.1200 | 0.0620 | 0.0620 | 0.0840 | 0.3500 | 0.2620 | 0.9120 | 0.3500 | 0.6216 |
| clean_boundary_active__step_0.0025 | 0.2160 | 0.1840 | 0.1220 | 0.0600 | 0.0600 | 0.0840 | 0.3600 | 0.2620 | 0.9120 | 0.3480 | 0.6204 |
| clean_boundary_active__step_0.005 | 0.2180 | 0.1860 | 0.1240 | 0.0600 | 0.0600 | 0.0820 | 0.3500 | 0.2620 | 0.9120 | 0.3460 | 0.6180 |
| clean_boundary_active__step_0.01 | 0.2200 | 0.1860 | 0.1240 | 0.0600 | 0.0600 | 0.0820 | 0.3560 | 0.2620 | 0.9120 | 0.3420 | 0.6136 |
| clean_boundary_active__step_0.02 | 0.2260 | 0.1900 | 0.1260 | 0.0620 | 0.0620 | 0.0840 | 0.3520 | 0.2660 | 0.9120 | 0.3440 | 0.6052 |
| clean_boundary_active__step_0.04 | 0.2260 | 0.1920 | 0.1360 | 0.0620 | 0.0620 | 0.0720 | 0.3620 | 0.2740 | 0.9160 | 0.3300 | 0.5984 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.2160 | 0.1840 | 0.1200 | 0.0620 | 0.0620 | 0.0840 | 0.3540 | 0.2620 | 0.9120 | 0.3520 | 0.6220 |
| cohen_aligned_noisy_margin__step_0.005 | 0.2160 | 0.1860 | 0.1240 | 0.0620 | 0.0620 | 0.0860 | 0.3540 | 0.2620 | 0.9120 | 0.3520 | 0.6220 |
| cohen_aligned_noisy_margin__step_0.01 | 0.2160 | 0.1860 | 0.1300 | 0.0620 | 0.0620 | 0.0840 | 0.3600 | 0.2620 | 0.9120 | 0.3560 | 0.6192 |
| cohen_aligned_noisy_margin__step_0.02 | 0.2140 | 0.1820 | 0.1300 | 0.0620 | 0.0620 | 0.0880 | 0.3660 | 0.2640 | 0.9120 | 0.3640 | 0.6168 |
| cohen_aligned_noisy_margin__step_0.04 | 0.2140 | 0.1980 | 0.1360 | 0.0640 | 0.0640 | 0.0900 | 0.3640 | 0.2740 | 0.9120 | 0.3900 | 0.6084 |
| control__learned_shared_translation_pure | 0.1920 | 0.1720 | 0.1280 | 0.0560 | 0.0560 | 0.0740 | 0.2940 | 0.2400 | 0.9100 | 0.3980 | 0.6752 |
| control__learned_shared_translation_tangent | 0.2300 | 0.1880 | 0.1420 | 0.0800 | 0.0780 | 0.0960 | 0.3160 | 0.2940 | 0.9200 | 0.4660 | 0.6092 |
| control__lowrank_tangent_r8 | 0.3720 | 0.3160 | 0.2540 | 0.1680 | 0.1620 | 0.0840 | 0.2880 | 0.4440 | 0.9240 | 0.1440 | 0.1196 |
| control__noisy_class_mean | 0.2740 | 0.2120 | 0.1600 | 0.0960 | 0.0660 | 0.1020 | 0.3260 | 0.3620 | 0.3740 | 0.1840 | 0.2476 |
| global_mean_centering__coefficient_0.25 | 0.2120 | 0.1880 | 0.1220 | 0.0660 | 0.0660 | 0.0980 | 0.3220 | 0.2460 | 0.9100 | 0.4040 | 0.6436 |
| global_mean_centering__coefficient_0.5 | 0.2080 | 0.1680 | 0.1300 | 0.0720 | 0.0720 | 0.1220 | 0.2540 | 0.2460 | 0.8840 | 0.4960 | 0.6844 |
| global_mean_centering__coefficient_1 | 0.1820 | 0.1500 | 0.1240 | 0.0800 | 0.0760 | 0.2000 | 0.2240 | 0.2220 | 0.8400 | 0.6240 | 0.7340 |
| gr_clip_style_two_sided__coefficient_1 | 0.2240 | 0.1920 | 0.1500 | 0.0700 | 0.0700 | 0.0920 | 0.2940 | 0.2620 | 0.9160 | 0.4280 | 0.6836 |
| no_correction | 0.2180 | 0.1800 | 0.1200 | 0.0620 | 0.0620 | 0.0840 | 0.3500 | 0.2620 | 0.9120 | 0.3500 | 0.6216 |

#### openai-clip-vit-b16-quickgelu__eurosat__sigma0.25 (sigma 0.25, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1000 | 0.0733 | 0.0467 | 0.0200 | 0.0267 | 0.4533 | 0.1533 | 0.1267 | 0.5667 | 0.8067 | 0.8147 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1000 | 0.0733 | 0.0467 | 0.0200 | 0.0267 | 0.4533 | 0.1533 | 0.1267 | 0.5667 | 0.8067 | 0.8147 |
| chowers_exact_projected_gap__coefficient_1 | 0.1000 | 0.0733 | 0.0467 | 0.0200 | 0.0267 | 0.4533 | 0.1533 | 0.1267 | 0.5667 | 0.8067 | 0.8147 |
| clean_boundary_active__step_0.0025 | 0.1000 | 0.0733 | 0.0533 | 0.0200 | 0.0333 | 0.4400 | 0.1533 | 0.1267 | 0.5667 | 0.8067 | 0.8213 |
| clean_boundary_active__step_0.005 | 0.0867 | 0.0800 | 0.0533 | 0.0200 | 0.0333 | 0.4333 | 0.1733 | 0.1267 | 0.5667 | 0.8067 | 0.8213 |
| clean_boundary_active__step_0.01 | 0.1000 | 0.0800 | 0.0600 | 0.0200 | 0.0400 | 0.4400 | 0.1467 | 0.1267 | 0.5667 | 0.7800 | 0.8160 |
| clean_boundary_active__step_0.02 | 0.1000 | 0.0800 | 0.0600 | 0.0133 | 0.0400 | 0.4333 | 0.1600 | 0.1333 | 0.5600 | 0.7667 | 0.8107 |
| clean_boundary_active__step_0.04 | 0.1267 | 0.0800 | 0.0600 | 0.0200 | 0.0267 | 0.4200 | 0.1200 | 0.1533 | 0.5467 | 0.7267 | 0.7947 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1000 | 0.0733 | 0.0533 | 0.0200 | 0.0333 | 0.4467 | 0.1600 | 0.1267 | 0.5667 | 0.8067 | 0.8147 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1000 | 0.0800 | 0.0533 | 0.0200 | 0.0333 | 0.4467 | 0.1600 | 0.1267 | 0.5667 | 0.8133 | 0.8200 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1000 | 0.0800 | 0.0600 | 0.0200 | 0.0400 | 0.4533 | 0.1467 | 0.1200 | 0.5667 | 0.7933 | 0.8213 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1000 | 0.0800 | 0.0600 | 0.0133 | 0.0400 | 0.4600 | 0.1200 | 0.1267 | 0.5533 | 0.7933 | 0.8240 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1067 | 0.0800 | 0.0667 | 0.0200 | 0.0333 | 0.4667 | 0.1333 | 0.1267 | 0.5400 | 0.7800 | 0.8280 |
| control__learned_shared_translation_pure | 0.1067 | 0.1067 | 0.0867 | 0.0400 | 0.0800 | 0.6267 | 0.0467 | 0.1133 | 0.3733 | 0.9133 | 0.8720 |
| control__learned_shared_translation_tangent | 0.1600 | 0.0800 | 0.0467 | 0.0067 | 0.0200 | 0.1000 | 0.3333 | 0.2333 | 0.4400 | 0.2067 | 0.3840 |
| control__lowrank_tangent_r8 | 0.5600 | 0.5067 | 0.4200 | 0.2533 | 0.3600 | 0.1667 | 0.0933 | 0.5800 | 0.8000 | 0.1800 | 0.2987 |
| control__noisy_class_mean | 0.5467 | 0.4533 | 0.3400 | 0.1400 | 0.1133 | 0.0867 | 0.1467 | 0.5800 | 0.1733 | 0.1667 | 0.2760 |
| global_mean_centering__coefficient_0.25 | 0.1067 | 0.0933 | 0.0600 | 0.0333 | 0.0467 | 0.5533 | 0.0667 | 0.1133 | 0.5533 | 0.9067 | 0.8613 |
| global_mean_centering__coefficient_0.5 | 0.0933 | 0.0933 | 0.0933 | 0.0667 | 0.0867 | 0.7400 | 0.0200 | 0.1000 | 0.5667 | 0.9733 | 0.8867 |
| global_mean_centering__coefficient_1 | 0.1000 | 0.1000 | 0.1000 | 0.0933 | 0.0733 | 0.8400 | 0.0133 | 0.1000 | 0.5067 | 1.0000 | 0.9000 |
| gr_clip_style_two_sided__coefficient_1 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.8800 | 0.0000 | 0.1000 | 0.6200 | 1.0000 | 0.9000 |
| no_correction | 0.1000 | 0.0733 | 0.0467 | 0.0200 | 0.0267 | 0.4533 | 0.1533 | 0.1267 | 0.5667 | 0.8067 | 0.8147 |

#### openai-clip-vit-b16-quickgelu__eurosat__sigma0.5 (sigma 0.5, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1200 | 0.1067 | 0.0933 | 0.0533 | 0.0467 | 0.2200 | 0.2933 | 0.1467 | 0.5667 | 0.4333 | 0.7427 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1200 | 0.1067 | 0.0933 | 0.0533 | 0.0467 | 0.2200 | 0.2933 | 0.1467 | 0.5667 | 0.4333 | 0.7427 |
| chowers_exact_projected_gap__coefficient_1 | 0.1200 | 0.1067 | 0.0933 | 0.0533 | 0.0467 | 0.2200 | 0.2933 | 0.1467 | 0.5667 | 0.4333 | 0.7427 |
| clean_boundary_active__step_0.0025 | 0.1267 | 0.1067 | 0.0933 | 0.0533 | 0.0467 | 0.2200 | 0.2733 | 0.1533 | 0.5667 | 0.4667 | 0.7547 |
| clean_boundary_active__step_0.005 | 0.1200 | 0.1067 | 0.0933 | 0.0600 | 0.0533 | 0.2267 | 0.2933 | 0.1533 | 0.5667 | 0.5000 | 0.7667 |
| clean_boundary_active__step_0.01 | 0.1267 | 0.1067 | 0.0933 | 0.0600 | 0.0533 | 0.2533 | 0.2667 | 0.1533 | 0.5667 | 0.5600 | 0.7867 |
| clean_boundary_active__step_0.02 | 0.1333 | 0.1133 | 0.0933 | 0.0733 | 0.0600 | 0.3200 | 0.1800 | 0.1533 | 0.5600 | 0.6800 | 0.8200 |
| clean_boundary_active__step_0.04 | 0.1333 | 0.1200 | 0.1133 | 0.0800 | 0.0600 | 0.4600 | 0.0800 | 0.1400 | 0.5467 | 0.8000 | 0.8520 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1267 | 0.1067 | 0.0933 | 0.0533 | 0.0467 | 0.2200 | 0.2733 | 0.1533 | 0.5667 | 0.4667 | 0.7547 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1200 | 0.1067 | 0.0933 | 0.0600 | 0.0533 | 0.2267 | 0.2800 | 0.1600 | 0.5667 | 0.4933 | 0.7653 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1267 | 0.1067 | 0.0933 | 0.0600 | 0.0533 | 0.2467 | 0.2600 | 0.1467 | 0.5667 | 0.5400 | 0.7800 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1267 | 0.1200 | 0.0933 | 0.0733 | 0.0533 | 0.3200 | 0.2067 | 0.1600 | 0.5533 | 0.6867 | 0.8293 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1333 | 0.1200 | 0.1133 | 0.0867 | 0.0667 | 0.4267 | 0.0600 | 0.1400 | 0.5400 | 0.7800 | 0.8493 |
| control__learned_shared_translation_pure | 0.1267 | 0.1133 | 0.1067 | 0.0667 | 0.0533 | 0.4200 | 0.0933 | 0.1400 | 0.3600 | 0.7667 | 0.8413 |
| control__learned_shared_translation_tangent | 0.1067 | 0.0867 | 0.0667 | 0.0467 | 0.0267 | 0.1067 | 0.3800 | 0.1333 | 0.4867 | 0.4200 | 0.6960 |
| control__lowrank_tangent_r8 | 0.4800 | 0.4400 | 0.3733 | 0.3200 | 0.3000 | 0.1333 | 0.2000 | 0.5533 | 0.8400 | 0.2067 | 0.3947 |
| control__noisy_class_mean | 0.4600 | 0.3667 | 0.2667 | 0.1267 | 0.0133 | 0.0667 | 0.2333 | 0.5133 | 0.1533 | 0.2200 | 0.3680 |
| global_mean_centering__coefficient_0.25 | 0.0933 | 0.0933 | 0.0800 | 0.0667 | 0.0533 | 0.2733 | 0.1800 | 0.1267 | 0.5533 | 0.8333 | 0.8480 |
| global_mean_centering__coefficient_0.5 | 0.0800 | 0.0800 | 0.0800 | 0.0733 | 0.0667 | 0.7067 | 0.0600 | 0.0933 | 0.5667 | 0.9600 | 0.8907 |
| global_mean_centering__coefficient_1 | 0.0933 | 0.0867 | 0.0800 | 0.0733 | 0.0533 | 0.7533 | 0.0200 | 0.0933 | 0.5067 | 0.9400 | 0.8867 |
| gr_clip_style_two_sided__coefficient_1 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.9000 | 0.0000 | 0.1000 | 0.6200 | 1.0000 | 0.9000 |
| no_correction | 0.1200 | 0.1067 | 0.0933 | 0.0533 | 0.0467 | 0.2200 | 0.2933 | 0.1467 | 0.5667 | 0.4333 | 0.7427 |

#### openclip-vit-b32-laion2b__cifar100__sigma0.25 (sigma 0.25, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1160 | 0.0900 | 0.0680 | 0.0400 | 0.0600 | 0.3700 | 0.1540 | 0.1320 | 0.7540 | 0.5700 | 0.9042 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1160 | 0.0900 | 0.0680 | 0.0400 | 0.0600 | 0.3720 | 0.1540 | 0.1320 | 0.7540 | 0.5700 | 0.9042 |
| chowers_exact_projected_gap__coefficient_1 | 0.1160 | 0.0900 | 0.0680 | 0.0400 | 0.0600 | 0.3720 | 0.1540 | 0.1320 | 0.7540 | 0.5700 | 0.9042 |
| clean_boundary_active__step_0.0025 | 0.1160 | 0.0920 | 0.0680 | 0.0400 | 0.0600 | 0.3720 | 0.1520 | 0.1320 | 0.7540 | 0.5700 | 0.9042 |
| clean_boundary_active__step_0.005 | 0.1160 | 0.0920 | 0.0680 | 0.0400 | 0.0600 | 0.3740 | 0.1480 | 0.1320 | 0.7540 | 0.5720 | 0.9042 |
| clean_boundary_active__step_0.01 | 0.1160 | 0.0920 | 0.0680 | 0.0400 | 0.0600 | 0.3740 | 0.1480 | 0.1320 | 0.7540 | 0.5720 | 0.9025 |
| clean_boundary_active__step_0.02 | 0.1160 | 0.0920 | 0.0680 | 0.0400 | 0.0600 | 0.3840 | 0.1480 | 0.1320 | 0.7540 | 0.5800 | 0.9039 |
| clean_boundary_active__step_0.04 | 0.1160 | 0.0920 | 0.0700 | 0.0420 | 0.0620 | 0.3960 | 0.1540 | 0.1340 | 0.7540 | 0.5960 | 0.9059 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1160 | 0.0920 | 0.0680 | 0.0400 | 0.0600 | 0.3720 | 0.1560 | 0.1320 | 0.7540 | 0.5680 | 0.9041 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1160 | 0.0920 | 0.0680 | 0.0400 | 0.0600 | 0.3680 | 0.1560 | 0.1320 | 0.7540 | 0.5640 | 0.9039 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1160 | 0.0920 | 0.0680 | 0.0400 | 0.0600 | 0.3640 | 0.1560 | 0.1320 | 0.7540 | 0.5640 | 0.9039 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1160 | 0.0920 | 0.0680 | 0.0400 | 0.0600 | 0.3640 | 0.1620 | 0.1320 | 0.7540 | 0.5640 | 0.9039 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1160 | 0.0940 | 0.0680 | 0.0400 | 0.0600 | 0.3580 | 0.1680 | 0.1320 | 0.7540 | 0.5600 | 0.9032 |
| control__learned_shared_translation_pure | 0.0980 | 0.0780 | 0.0500 | 0.0320 | 0.0460 | 0.3980 | 0.1480 | 0.1060 | 0.7180 | 0.5740 | 0.9334 |
| control__learned_shared_translation_tangent | 0.1380 | 0.1140 | 0.0820 | 0.0500 | 0.0780 | 0.2780 | 0.2020 | 0.1580 | 0.7320 | 0.4360 | 0.8813 |
| control__lowrank_tangent_r8 | 0.2380 | 0.1960 | 0.1380 | 0.0760 | 0.1260 | 0.0880 | 0.3780 | 0.2860 | 0.7640 | 0.0380 | 0.4226 |
| control__noisy_class_mean | 0.2240 | 0.1780 | 0.1340 | 0.0700 | 0.0980 | 0.0900 | 0.3660 | 0.2840 | 0.3040 | 0.0380 | 0.3614 |
| global_mean_centering__coefficient_0.25 | 0.1220 | 0.1000 | 0.0720 | 0.0380 | 0.0620 | 0.3840 | 0.1660 | 0.1380 | 0.7560 | 0.5660 | 0.9038 |
| global_mean_centering__coefficient_0.5 | 0.1300 | 0.1060 | 0.0740 | 0.0440 | 0.0640 | 0.3500 | 0.1720 | 0.1440 | 0.7540 | 0.5280 | 0.9006 |
| global_mean_centering__coefficient_1 | 0.1220 | 0.1060 | 0.0740 | 0.0400 | 0.0700 | 0.3120 | 0.2100 | 0.1440 | 0.7140 | 0.4140 | 0.9060 |
| gr_clip_style_two_sided__coefficient_1 | 0.1420 | 0.1140 | 0.0800 | 0.0500 | 0.0700 | 0.2260 | 0.2800 | 0.1660 | 0.7580 | 0.3100 | 0.8761 |
| no_correction | 0.1160 | 0.0900 | 0.0680 | 0.0400 | 0.0600 | 0.3720 | 0.1540 | 0.1320 | 0.7540 | 0.5700 | 0.9042 |

#### openclip-vit-b32-laion2b__cifar100__sigma0.5 (sigma 0.5, 500 items, 100 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0400 | 0.0340 | 0.0280 | 0.0180 | 0.0120 | 0.5000 | 0.1360 | 0.0500 | 0.7540 | 0.7860 | 0.9674 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0400 | 0.0340 | 0.0280 | 0.0180 | 0.0120 | 0.5000 | 0.1360 | 0.0500 | 0.7540 | 0.7860 | 0.9674 |
| chowers_exact_projected_gap__coefficient_1 | 0.0400 | 0.0340 | 0.0280 | 0.0180 | 0.0120 | 0.5000 | 0.1360 | 0.0500 | 0.7540 | 0.7860 | 0.9674 |
| clean_boundary_active__step_0.0025 | 0.0400 | 0.0340 | 0.0280 | 0.0180 | 0.0120 | 0.5060 | 0.1360 | 0.0500 | 0.7540 | 0.7880 | 0.9676 |
| clean_boundary_active__step_0.005 | 0.0400 | 0.0320 | 0.0280 | 0.0180 | 0.0120 | 0.5060 | 0.1380 | 0.0500 | 0.7540 | 0.7880 | 0.9676 |
| clean_boundary_active__step_0.01 | 0.0400 | 0.0320 | 0.0300 | 0.0180 | 0.0120 | 0.5100 | 0.1400 | 0.0480 | 0.7540 | 0.7900 | 0.9680 |
| clean_boundary_active__step_0.02 | 0.0400 | 0.0340 | 0.0280 | 0.0180 | 0.0120 | 0.5240 | 0.1400 | 0.0480 | 0.7540 | 0.7980 | 0.9676 |
| clean_boundary_active__step_0.04 | 0.0400 | 0.0340 | 0.0260 | 0.0160 | 0.0100 | 0.5460 | 0.1260 | 0.0460 | 0.7540 | 0.8060 | 0.9681 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0400 | 0.0340 | 0.0300 | 0.0180 | 0.0120 | 0.4940 | 0.1320 | 0.0500 | 0.7540 | 0.7840 | 0.9673 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0400 | 0.0340 | 0.0300 | 0.0180 | 0.0120 | 0.4940 | 0.1320 | 0.0500 | 0.7520 | 0.7840 | 0.9673 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0400 | 0.0340 | 0.0320 | 0.0180 | 0.0120 | 0.4940 | 0.1320 | 0.0500 | 0.7520 | 0.7860 | 0.9682 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0400 | 0.0340 | 0.0320 | 0.0180 | 0.0120 | 0.4940 | 0.1320 | 0.0520 | 0.7520 | 0.7840 | 0.9682 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0400 | 0.0380 | 0.0320 | 0.0180 | 0.0120 | 0.4760 | 0.1360 | 0.0540 | 0.7540 | 0.7800 | 0.9672 |
| control__learned_shared_translation_pure | 0.0340 | 0.0260 | 0.0100 | 0.0060 | 0.0060 | 0.1500 | 0.3180 | 0.0360 | 0.6960 | 0.3640 | 0.9579 |
| control__learned_shared_translation_tangent | 0.0400 | 0.0360 | 0.0300 | 0.0180 | 0.0140 | 0.4880 | 0.1640 | 0.0520 | 0.7360 | 0.7960 | 0.9682 |
| control__lowrank_tangent_r8 | 0.1060 | 0.0800 | 0.0620 | 0.0460 | 0.0420 | 0.0340 | 0.6140 | 0.1460 | 0.7780 | 0.0760 | 0.6618 |
| control__noisy_class_mean | 0.0720 | 0.0560 | 0.0500 | 0.0240 | 0.0120 | 0.0560 | 0.6680 | 0.1520 | 0.1340 | 0.1320 | 0.5866 |
| global_mean_centering__coefficient_0.25 | 0.0380 | 0.0360 | 0.0260 | 0.0200 | 0.0140 | 0.5080 | 0.1260 | 0.0420 | 0.7560 | 0.8080 | 0.9699 |
| global_mean_centering__coefficient_0.5 | 0.0340 | 0.0340 | 0.0320 | 0.0180 | 0.0140 | 0.4380 | 0.1820 | 0.0420 | 0.7540 | 0.7720 | 0.9677 |
| global_mean_centering__coefficient_1 | 0.0440 | 0.0400 | 0.0300 | 0.0220 | 0.0200 | 0.1700 | 0.3820 | 0.0520 | 0.7140 | 0.4800 | 0.9475 |
| gr_clip_style_two_sided__coefficient_1 | 0.0300 | 0.0280 | 0.0260 | 0.0180 | 0.0180 | 0.1800 | 0.2700 | 0.0400 | 0.7580 | 0.6020 | 0.9591 |
| no_correction | 0.0400 | 0.0340 | 0.0280 | 0.0180 | 0.0120 | 0.5000 | 0.1360 | 0.0500 | 0.7540 | 0.7860 | 0.9674 |

#### openclip-vit-b32-laion2b__cifar10__sigma0.25 (sigma 0.25, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.3680 | 0.2960 | 0.2280 | 0.1200 | 0.2260 | 0.1400 | 0.1720 | 0.4160 | 0.9360 | 0.2980 | 0.5052 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.3680 | 0.2960 | 0.2280 | 0.1200 | 0.2260 | 0.1400 | 0.1720 | 0.4160 | 0.9360 | 0.2980 | 0.5052 |
| chowers_exact_projected_gap__coefficient_1 | 0.3680 | 0.2960 | 0.2280 | 0.1200 | 0.2260 | 0.1400 | 0.1720 | 0.4160 | 0.9360 | 0.2980 | 0.5052 |
| clean_boundary_active__step_0.0025 | 0.3680 | 0.2980 | 0.2260 | 0.1220 | 0.2240 | 0.1400 | 0.1780 | 0.4160 | 0.9360 | 0.2980 | 0.5048 |
| clean_boundary_active__step_0.005 | 0.3720 | 0.3000 | 0.2260 | 0.1220 | 0.2240 | 0.1420 | 0.1740 | 0.4180 | 0.9360 | 0.2980 | 0.5024 |
| clean_boundary_active__step_0.01 | 0.3720 | 0.3000 | 0.2280 | 0.1220 | 0.2260 | 0.1340 | 0.1780 | 0.4260 | 0.9360 | 0.2920 | 0.4960 |
| clean_boundary_active__step_0.02 | 0.3800 | 0.3060 | 0.2280 | 0.1200 | 0.2260 | 0.1300 | 0.1700 | 0.4280 | 0.9360 | 0.2840 | 0.4884 |
| clean_boundary_active__step_0.04 | 0.3980 | 0.3080 | 0.2280 | 0.1220 | 0.2240 | 0.1240 | 0.1780 | 0.4480 | 0.9360 | 0.2680 | 0.4616 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.3680 | 0.3000 | 0.2280 | 0.1220 | 0.2260 | 0.1400 | 0.1740 | 0.4160 | 0.9360 | 0.2980 | 0.5052 |
| cohen_aligned_noisy_margin__step_0.005 | 0.3680 | 0.3000 | 0.2280 | 0.1220 | 0.2260 | 0.1400 | 0.1760 | 0.4160 | 0.9360 | 0.2980 | 0.5052 |
| cohen_aligned_noisy_margin__step_0.01 | 0.3720 | 0.3000 | 0.2260 | 0.1240 | 0.2240 | 0.1380 | 0.1760 | 0.4200 | 0.9360 | 0.2980 | 0.5028 |
| cohen_aligned_noisy_margin__step_0.02 | 0.3720 | 0.3000 | 0.2300 | 0.1280 | 0.2280 | 0.1440 | 0.1800 | 0.4320 | 0.9360 | 0.2940 | 0.4964 |
| cohen_aligned_noisy_margin__step_0.04 | 0.3840 | 0.3040 | 0.2320 | 0.1260 | 0.2300 | 0.1320 | 0.1640 | 0.4420 | 0.9340 | 0.2900 | 0.4860 |
| control__learned_shared_translation_pure | 0.3260 | 0.2600 | 0.2100 | 0.1020 | 0.2080 | 0.2300 | 0.1260 | 0.3720 | 0.9380 | 0.4080 | 0.6072 |
| control__learned_shared_translation_tangent | 0.4660 | 0.3420 | 0.2660 | 0.1240 | 0.2640 | 0.0580 | 0.2280 | 0.5440 | 0.9340 | 0.1720 | 0.2540 |
| control__lowrank_tangent_r8 | 0.5720 | 0.4940 | 0.3920 | 0.2160 | 0.3860 | 0.1000 | 0.1100 | 0.5980 | 0.9280 | 0.1320 | 0.0992 |
| control__noisy_class_mean | 0.4480 | 0.3760 | 0.2880 | 0.1560 | 0.2020 | 0.1480 | 0.1380 | 0.4900 | 0.5080 | 0.1340 | 0.1500 |
| global_mean_centering__coefficient_0.25 | 0.3700 | 0.2960 | 0.2260 | 0.1240 | 0.2240 | 0.1420 | 0.1660 | 0.4080 | 0.9360 | 0.3240 | 0.5232 |
| global_mean_centering__coefficient_0.5 | 0.3640 | 0.2940 | 0.2240 | 0.1220 | 0.2220 | 0.1460 | 0.1680 | 0.4040 | 0.9360 | 0.3400 | 0.5576 |
| global_mean_centering__coefficient_1 | 0.3680 | 0.2980 | 0.2220 | 0.1200 | 0.2220 | 0.1320 | 0.1640 | 0.4040 | 0.9340 | 0.3320 | 0.5464 |
| gr_clip_style_two_sided__coefficient_1 | 0.4500 | 0.3580 | 0.2460 | 0.1020 | 0.2400 | 0.0720 | 0.2260 | 0.5160 | 0.9400 | 0.2280 | 0.2916 |
| no_correction | 0.3680 | 0.2960 | 0.2280 | 0.1200 | 0.2260 | 0.1400 | 0.1720 | 0.4160 | 0.9360 | 0.2980 | 0.5052 |

#### openclip-vit-b32-laion2b__cifar10__sigma0.5 (sigma 0.5, 500 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1380 | 0.1240 | 0.0960 | 0.0700 | 0.0660 | 0.2620 | 0.1580 | 0.1540 | 0.9360 | 0.7080 | 0.7988 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1380 | 0.1240 | 0.0960 | 0.0700 | 0.0660 | 0.2620 | 0.1580 | 0.1540 | 0.9360 | 0.7080 | 0.7988 |
| chowers_exact_projected_gap__coefficient_1 | 0.1380 | 0.1240 | 0.0960 | 0.0700 | 0.0660 | 0.2620 | 0.1580 | 0.1540 | 0.9360 | 0.7080 | 0.7988 |
| clean_boundary_active__step_0.0025 | 0.1360 | 0.1240 | 0.0960 | 0.0680 | 0.0640 | 0.2560 | 0.1640 | 0.1560 | 0.9360 | 0.7020 | 0.7960 |
| clean_boundary_active__step_0.005 | 0.1360 | 0.1260 | 0.1000 | 0.0680 | 0.0640 | 0.2520 | 0.1640 | 0.1560 | 0.9360 | 0.7000 | 0.7948 |
| clean_boundary_active__step_0.01 | 0.1380 | 0.1240 | 0.1000 | 0.0620 | 0.0580 | 0.2380 | 0.1700 | 0.1580 | 0.9360 | 0.6900 | 0.7880 |
| clean_boundary_active__step_0.02 | 0.1440 | 0.1260 | 0.0960 | 0.0580 | 0.0540 | 0.2100 | 0.1660 | 0.1600 | 0.9360 | 0.6700 | 0.7764 |
| clean_boundary_active__step_0.04 | 0.1420 | 0.1240 | 0.0960 | 0.0560 | 0.0520 | 0.1600 | 0.2040 | 0.1640 | 0.9360 | 0.6260 | 0.7576 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.1380 | 0.1260 | 0.0960 | 0.0700 | 0.0660 | 0.2580 | 0.1580 | 0.1540 | 0.9360 | 0.7060 | 0.7984 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1380 | 0.1260 | 0.1000 | 0.0700 | 0.0660 | 0.2560 | 0.1580 | 0.1560 | 0.9360 | 0.7020 | 0.7968 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1380 | 0.1240 | 0.1000 | 0.0660 | 0.0640 | 0.2540 | 0.1640 | 0.1560 | 0.9360 | 0.7020 | 0.7960 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1460 | 0.1240 | 0.1040 | 0.0640 | 0.0620 | 0.2300 | 0.1620 | 0.1640 | 0.9360 | 0.6920 | 0.7888 |
| cohen_aligned_noisy_margin__step_0.04 | 0.1500 | 0.1240 | 0.1020 | 0.0560 | 0.0520 | 0.1940 | 0.1700 | 0.1700 | 0.9340 | 0.6640 | 0.7780 |
| control__learned_shared_translation_pure | 0.1160 | 0.1080 | 0.1040 | 0.0880 | 0.0860 | 0.5280 | 0.0660 | 0.1200 | 0.9360 | 0.8660 | 0.8624 |
| control__learned_shared_translation_tangent | 0.1740 | 0.1440 | 0.1000 | 0.0560 | 0.0540 | 0.0460 | 0.4380 | 0.2760 | 0.9300 | 0.3400 | 0.5528 |
| control__lowrank_tangent_r8 | 0.3180 | 0.2520 | 0.2000 | 0.1100 | 0.1080 | 0.0780 | 0.3220 | 0.4080 | 0.9280 | 0.1560 | 0.1572 |
| control__noisy_class_mean | 0.2000 | 0.1560 | 0.1360 | 0.0740 | 0.0380 | 0.1020 | 0.4160 | 0.3200 | 0.3380 | 0.1820 | 0.2308 |
| global_mean_centering__coefficient_0.25 | 0.1280 | 0.1200 | 0.1020 | 0.0720 | 0.0680 | 0.3160 | 0.1240 | 0.1480 | 0.9360 | 0.7560 | 0.8216 |
| global_mean_centering__coefficient_0.5 | 0.1320 | 0.1200 | 0.1060 | 0.0720 | 0.0680 | 0.3820 | 0.0720 | 0.1400 | 0.9360 | 0.7920 | 0.8452 |
| global_mean_centering__coefficient_1 | 0.1320 | 0.1160 | 0.1000 | 0.0700 | 0.0680 | 0.3420 | 0.0900 | 0.1400 | 0.9340 | 0.7860 | 0.8432 |
| gr_clip_style_two_sided__coefficient_1 | 0.1440 | 0.1240 | 0.1000 | 0.0620 | 0.0600 | 0.2200 | 0.2140 | 0.1800 | 0.9400 | 0.6400 | 0.7640 |
| no_correction | 0.1380 | 0.1240 | 0.0960 | 0.0700 | 0.0660 | 0.2620 | 0.1580 | 0.1540 | 0.9360 | 0.7080 | 0.7988 |

#### openclip-vit-b32-laion2b__eurosat__sigma0.25 (sigma 0.25, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.0933 | 0.0800 | 0.0533 | 0.0267 | 0.0067 | 0.4333 | 0.1200 | 0.1267 | 0.5000 | 0.7067 | 0.8027 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.0933 | 0.0800 | 0.0533 | 0.0267 | 0.0067 | 0.4333 | 0.1200 | 0.1267 | 0.5000 | 0.7067 | 0.8027 |
| chowers_exact_projected_gap__coefficient_1 | 0.0933 | 0.0800 | 0.0533 | 0.0267 | 0.0067 | 0.4333 | 0.1200 | 0.1267 | 0.5000 | 0.7067 | 0.8027 |
| clean_boundary_active__step_0.0025 | 0.1000 | 0.0867 | 0.0533 | 0.0200 | 0.0067 | 0.4333 | 0.1333 | 0.1267 | 0.5000 | 0.6800 | 0.7960 |
| clean_boundary_active__step_0.005 | 0.1000 | 0.0933 | 0.0533 | 0.0200 | 0.0067 | 0.4133 | 0.1467 | 0.1200 | 0.5000 | 0.6667 | 0.7920 |
| clean_boundary_active__step_0.01 | 0.1000 | 0.0800 | 0.0533 | 0.0133 | 0.0067 | 0.3933 | 0.1467 | 0.1200 | 0.5000 | 0.6533 | 0.7893 |
| clean_boundary_active__step_0.02 | 0.1067 | 0.0733 | 0.0400 | 0.0200 | 0.0133 | 0.3733 | 0.1267 | 0.1200 | 0.5000 | 0.5800 | 0.7693 |
| clean_boundary_active__step_0.04 | 0.1000 | 0.0600 | 0.0467 | 0.0200 | 0.0267 | 0.3267 | 0.1733 | 0.1000 | 0.5000 | 0.4533 | 0.7333 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0933 | 0.0800 | 0.0533 | 0.0267 | 0.0067 | 0.4333 | 0.1267 | 0.1267 | 0.5000 | 0.6933 | 0.7987 |
| cohen_aligned_noisy_margin__step_0.005 | 0.1000 | 0.0800 | 0.0533 | 0.0267 | 0.0067 | 0.4333 | 0.1267 | 0.1267 | 0.5000 | 0.6933 | 0.7987 |
| cohen_aligned_noisy_margin__step_0.01 | 0.1000 | 0.0867 | 0.0533 | 0.0267 | 0.0067 | 0.4333 | 0.1400 | 0.1200 | 0.5000 | 0.6733 | 0.7933 |
| cohen_aligned_noisy_margin__step_0.02 | 0.1133 | 0.0733 | 0.0533 | 0.0200 | 0.0067 | 0.3867 | 0.1000 | 0.1200 | 0.4867 | 0.6667 | 0.7893 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0933 | 0.0600 | 0.0467 | 0.0200 | 0.0067 | 0.3933 | 0.1533 | 0.1200 | 0.4800 | 0.6200 | 0.7747 |
| control__learned_shared_translation_pure | 0.1000 | 0.0867 | 0.0867 | 0.0667 | 0.0400 | 0.6667 | 0.0400 | 0.1000 | 0.4000 | 0.9267 | 0.8720 |
| control__learned_shared_translation_tangent | 0.2000 | 0.1267 | 0.0867 | 0.0400 | 0.0800 | 0.3733 | 0.1467 | 0.2267 | 0.4400 | 0.6400 | 0.7667 |
| control__lowrank_tangent_r8 | 0.6200 | 0.4867 | 0.4133 | 0.1933 | 0.3933 | 0.0933 | 0.1133 | 0.6733 | 0.8600 | 0.1533 | 0.1840 |
| control__noisy_class_mean | 0.4333 | 0.3467 | 0.2533 | 0.1400 | 0.0800 | 0.1467 | 0.1667 | 0.4667 | 0.1600 | 0.1400 | 0.1493 |
| global_mean_centering__coefficient_0.25 | 0.1000 | 0.0800 | 0.0733 | 0.0333 | 0.0067 | 0.4667 | 0.0867 | 0.1200 | 0.4933 | 0.7867 | 0.8347 |
| global_mean_centering__coefficient_0.5 | 0.1000 | 0.0800 | 0.0733 | 0.0533 | 0.0000 | 0.5133 | 0.0400 | 0.1000 | 0.4733 | 0.8800 | 0.8533 |
| global_mean_centering__coefficient_1 | 0.0867 | 0.0733 | 0.0667 | 0.0533 | 0.0067 | 0.5933 | 0.0800 | 0.0933 | 0.4800 | 0.8867 | 0.8707 |
| gr_clip_style_two_sided__coefficient_1 | 0.0400 | 0.0333 | 0.0200 | 0.0000 | 0.0200 | 0.2667 | 0.2467 | 0.0533 | 0.5733 | 0.5200 | 0.6960 |
| no_correction | 0.0933 | 0.0800 | 0.0533 | 0.0267 | 0.0067 | 0.4333 | 0.1200 | 0.1267 | 0.5000 | 0.7067 | 0.8027 |

#### openclip-vit-b32-laion2b__eurosat__sigma0.5 (sigma 0.5, 150 items, 10 classes, primary)

| bank | CA@0 | CA@0.12 | CA@0.25 | CA@0.5 | anchored@sigma | cert-wrong@sigma | abstention | smoothed | clean | top share | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0000 | 0.2933 | 0.0933 | 0.1000 | 0.5000 | 0.9000 | 0.8747 |
| chowers_exact_projected_gap__coefficient_0.5 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0000 | 0.2933 | 0.0933 | 0.1000 | 0.5000 | 0.9000 | 0.8747 |
| chowers_exact_projected_gap__coefficient_1 | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0000 | 0.2933 | 0.0933 | 0.1000 | 0.5000 | 0.9000 | 0.8747 |
| clean_boundary_active__step_0.0025 | 0.0933 | 0.0933 | 0.0933 | 0.0733 | 0.0000 | 0.2667 | 0.1067 | 0.1000 | 0.5000 | 0.8733 | 0.8680 |
| clean_boundary_active__step_0.005 | 0.0933 | 0.0933 | 0.0867 | 0.0733 | 0.0000 | 0.2467 | 0.1133 | 0.1000 | 0.5000 | 0.8600 | 0.8653 |
| clean_boundary_active__step_0.01 | 0.0933 | 0.0933 | 0.0867 | 0.0733 | 0.0000 | 0.1800 | 0.1133 | 0.1000 | 0.5000 | 0.8000 | 0.8467 |
| clean_boundary_active__step_0.02 | 0.1067 | 0.0867 | 0.0733 | 0.0600 | 0.0000 | 0.1400 | 0.1733 | 0.1200 | 0.5000 | 0.7267 | 0.8320 |
| clean_boundary_active__step_0.04 | 0.1133 | 0.0867 | 0.0733 | 0.0200 | 0.0000 | 0.1200 | 0.3333 | 0.1467 | 0.5000 | 0.4733 | 0.7493 |
| cohen_aligned_noisy_margin__step_0.0025 | 0.0933 | 0.0933 | 0.0933 | 0.0800 | 0.0000 | 0.2867 | 0.1000 | 0.1000 | 0.5000 | 0.8933 | 0.8733 |
| cohen_aligned_noisy_margin__step_0.005 | 0.0933 | 0.0933 | 0.0933 | 0.0733 | 0.0000 | 0.2733 | 0.1000 | 0.1000 | 0.5000 | 0.8933 | 0.8747 |
| cohen_aligned_noisy_margin__step_0.01 | 0.0933 | 0.0933 | 0.0867 | 0.0733 | 0.0000 | 0.2667 | 0.1133 | 0.1000 | 0.5000 | 0.8800 | 0.8680 |
| cohen_aligned_noisy_margin__step_0.02 | 0.0933 | 0.0933 | 0.0867 | 0.0733 | 0.0000 | 0.2133 | 0.1200 | 0.0933 | 0.4867 | 0.8467 | 0.8613 |
| cohen_aligned_noisy_margin__step_0.04 | 0.0933 | 0.0867 | 0.0800 | 0.0533 | 0.0000 | 0.1400 | 0.1600 | 0.1067 | 0.4800 | 0.7667 | 0.8387 |
| control__learned_shared_translation_pure | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.0133 | 0.7267 | 0.0133 | 0.1000 | 0.3533 | 0.9933 | 0.8987 |
| control__learned_shared_translation_tangent | 0.1333 | 0.1333 | 0.1133 | 0.0467 | 0.0400 | 0.4267 | 0.1267 | 0.1400 | 0.4467 | 0.7867 | 0.8373 |
| control__lowrank_tangent_r8 | 0.5200 | 0.4800 | 0.4133 | 0.2200 | 0.2000 | 0.0600 | 0.1800 | 0.5800 | 0.8800 | 0.1867 | 0.2880 |
| control__noisy_class_mean | 0.3667 | 0.3067 | 0.2400 | 0.1400 | 0.0800 | 0.0733 | 0.3400 | 0.4733 | 0.1067 | 0.1867 | 0.2760 |
| global_mean_centering__coefficient_0.25 | 0.1000 | 0.1000 | 0.0933 | 0.0867 | 0.0000 | 0.4067 | 0.0467 | 0.1000 | 0.4933 | 0.9400 | 0.8800 |
| global_mean_centering__coefficient_0.5 | 0.0933 | 0.0933 | 0.0933 | 0.0933 | 0.0000 | 0.6533 | 0.0133 | 0.1000 | 0.4733 | 0.9667 | 0.8933 |
| global_mean_centering__coefficient_1 | 0.0933 | 0.0933 | 0.0933 | 0.0867 | 0.0133 | 0.7200 | 0.0133 | 0.0933 | 0.4800 | 0.8800 | 0.8760 |
| gr_clip_style_two_sided__coefficient_1 | 0.1000 | 0.1000 | 0.1000 | 0.0467 | 0.0467 | 0.6133 | 0.0533 | 0.1000 | 0.5733 | 0.8600 | 0.8587 |
| no_correction | 0.1000 | 0.0933 | 0.0933 | 0.0800 | 0.0000 | 0.2933 | 0.0933 | 0.1000 | 0.5000 | 0.9000 | 0.8747 |

### Comparison with the registered d1_07 outputs

Analysis directory `results/satml2027_local/analysis/EXP-20260920-019B`; tolerance 1e-09.

| output | rows compared | max abs deviation | within tolerance |
|---|---|---|---|
| cells_csv | 396 | 0.000e+00 | yes |
| concentration_csv | 396 | 0.000e+00 | yes |
| macro_csv | 22 | 0.000e+00 | yes |
| contrasts_json (point, lower, upper) | 21 | 1.388e-17 | yes |
| contrasts_json critical value | 1 | 0.000e+00 | yes |

d1_07 band metadata: {"confidence_level": 0.95, "critical_max_absolute_deviation": 0.015629629629629632, "method": "shared_item_multiplicity_dataset_class_stratified_max_absolute_deviation", "replicates": 100000, "seed": 2026091605}; seed match = True, replicates match = True

