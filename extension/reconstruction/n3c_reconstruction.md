# EXP019_RESULT_REVIEW_20260925_V1: independent reconstruction of N3C-20260920-V3

Generated 2026-09-25T19:02:02.789352+00:00 from `results/satml2027_downloads/EXP019_N3C_20260925T172040Z` (count shards, feature stores, registered bank archives; d1_08 not imported).

N3C issues no certificate, selects no candidate and enters no contrast family; the phrase "cannot be certified" is not used. P2, P6, P7, P9 are descriptive/structural.

- registration `96fd42f268d95b48b1cfbffb9afcd28c9dc4cb5c91ce46f17d4b53d75c0f2622` (match = True); alpha 0.001; draws 512; sigma 0.25; k_min recomputed 456 (registered 456)

## Registered rules

| prediction | registered rule | observed | outcome |
|---|---|---|---|
| P1 | zero exceptions (every applicable candidate, every cell) | 0 exceptions over 72 applicable (cell, candidate) pairs, 2955117 flips | conforms |
| P4a | mean absolute calibration error <= 0.05 on both CIFAR-100 cells | openai-clip-vit-b32-quickgelu__cifar100__sigma0.25: 0.0028, openai-clip-vit-b32-quickgelu__eurosat__sigma0.25: 0.0006, openai-clip-vit-l14-quickgelu__cifar100__sigma0.25: 0.0023, openai-clip-vit-l14-quickgelu__eurosat__sigma0.25: 0.0008 | holds |
| P5 | rho > 0 in at least 3 of 4 cells | openai-clip-vit-b32-quickgelu__cifar100__sigma0.25: +0.7408, openai-clip-vit-b32-quickgelu__eurosat__sigma0.25: +0.9666, openai-clip-vit-l14-quickgelu__cifar100__sigma0.25: +0.6179, openai-clip-vit-l14-quickgelu__eurosat__sigma0.25: +0.7333 | holds |
| P7 | reported (partition per candidate, cell and k_0 regime); no pass/fail | see per-cell tables | reported |
| P9 | reported (taxonomy per candidate and cell) | see per-cell tables | reported |

## openai-clip-vit-b32-quickgelu__cifar100__sigma0.25 (1000 items x 512 draws, 100 classes)

- bank archive `12df71dcdca9846852800f99995cac094bd3818b141e07fb4b65363088456077`: all tensor hashes recompute = True; shard candidate metadata equals bank metadata = True
- recomputed no_correction vote vectors equal the stored count shard on 998/1000 items; float64 re-scoring changes the base argmax on 2 of 512000 draws
- P5: rho = +0.7408, permutation p = 0.0000, hub class 87 (alignment rank 1), top-class share 0.3363
- P4a: MACE = 0.0028; bins (predicted -> observed): 0.000->0.000, 0.003->0.004, 0.043->0.041, 0.227->0.222, 0.532->0.523, 0.813->0.813, 0.958->0.967, 0.996->0.996, 1.000->1.000, 1.000->1.000
- P6: items already at k_min = 311; winner-runner-up margin quantiles q0.1=0.0008, q0.25=0.0021, q0.5=0.0053, q0.75=0.0115, q0.9=0.0202

### P1 containment and P7 flip budget

| candidate | applicable | flips | outside eligible | conforms | k0>=k_min | k0+E_F<k_min | undetermined | bound violations | E_F median | k1>=k_min |
|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | yes | 4 | 0 | yes | 311 | 689 | 0 | 0 | 0.0 | 311 |
| chowers_exact_projected_gap__coefficient_0.5 | yes | 4 | 0 | yes | 311 | 689 | 0 | 0 | 0.0 | 311 |
| chowers_exact_projected_gap__coefficient_1 | yes | 10 | 0 | yes | 311 | 689 | 0 | 0 | 0.0 | 311 |
| clean_boundary_active__step_0.0025 | yes | 2215 | 0 | yes | 311 | 671 | 18 | 0 | 8.0 | 313 |
| clean_boundary_active__step_0.005 | yes | 4424 | 0 | yes | 311 | 633 | 56 | 0 | 16.0 | 314 |
| clean_boundary_active__step_0.01 | yes | 8682 | 0 | yes | 311 | 525 | 164 | 0 | 34.0 | 313 |
| clean_boundary_active__step_0.02 | yes | 16410 | 0 | yes | 311 | 347 | 342 | 0 | 82.0 | 312 |
| clean_boundary_active__step_0.04 | yes | 30046 | 0 | yes | 311 | 101 | 588 | 0 | 267.0 | 312 |
| cohen_aligned_noisy_margin__step_0.0025 | yes | 1231 | 0 | yes | 311 | 680 | 9 | 0 | 6.0 | 311 |
| cohen_aligned_noisy_margin__step_0.005 | yes | 2444 | 0 | yes | 311 | 677 | 12 | 0 | 12.0 | 311 |
| cohen_aligned_noisy_margin__step_0.01 | yes | 4837 | 0 | yes | 311 | 641 | 48 | 0 | 24.0 | 309 |
| cohen_aligned_noisy_margin__step_0.02 | yes | 9550 | 0 | yes | 311 | 542 | 147 | 0 | 49.0 | 307 |
| cohen_aligned_noisy_margin__step_0.04 | yes | 19190 | 0 | yes | 311 | 303 | 386 | 0 | 100.5 | 307 |
| control__learned_shared_translation_pure | yes | 429440 | 0 | yes | 311 | 0 | 689 | 0 | 512.0 | 298 |
| control__learned_shared_translation_tangent | yes | 58993 | 0 | yes | 311 | 0 | 689 | 0 | 512.0 | 312 |
| control__lowrank_tangent_r8 | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |
| control__noisy_class_mean | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |
| global_mean_centering__coefficient_0.25 | yes | 101213 | 0 | yes | 311 | 4 | 685 | 0 | 491.0 | 290 |
| global_mean_centering__coefficient_0.5 | yes | 203844 | 0 | yes | 311 | 0 | 689 | 0 | 512.0 | 269 |
| global_mean_centering__coefficient_1 | yes | 328765 | 0 | yes | 311 | 0 | 689 | 0 | 512.0 | 255 |
| gr_clip_style_two_sided__coefficient_1 | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |

### P9 winner-movement taxonomy (paired draws)

| candidate | unchanged | useful wrong->correct | harmful correct->wrong | wrong->wrong | total flips |
|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 511996 | 1 | 0 | 3 | 4 |
| chowers_exact_projected_gap__coefficient_0.5 | 511996 | 0 | 0 | 4 | 4 |
| chowers_exact_projected_gap__coefficient_1 | 511990 | 0 | 1 | 9 | 10 |
| clean_boundary_active__step_0.0025 | 509785 | 207 | 140 | 1868 | 2215 |
| clean_boundary_active__step_0.005 | 507576 | 434 | 297 | 3693 | 4424 |
| clean_boundary_active__step_0.01 | 503318 | 866 | 611 | 7205 | 8682 |
| clean_boundary_active__step_0.02 | 495590 | 1679 | 1166 | 13565 | 16410 |
| clean_boundary_active__step_0.04 | 481954 | 3057 | 2253 | 24736 | 30046 |
| cohen_aligned_noisy_margin__step_0.0025 | 510769 | 60 | 138 | 1033 | 1231 |
| cohen_aligned_noisy_margin__step_0.005 | 509556 | 123 | 271 | 2050 | 2444 |
| cohen_aligned_noisy_margin__step_0.01 | 507163 | 250 | 581 | 4006 | 4837 |
| cohen_aligned_noisy_margin__step_0.02 | 502450 | 543 | 1104 | 7903 | 9550 |
| cohen_aligned_noisy_margin__step_0.04 | 492810 | 1113 | 2158 | 15919 | 19190 |
| control__learned_shared_translation_pure | 82560 | 13312 | 60387 | 355741 | 429440 |
| control__learned_shared_translation_tangent | 453007 | 6108 | 9831 | 43054 | 58993 |
| control__lowrank_tangent_r8 | 117146 | 73010 | 25318 | 296526 | 394854 |
| control__noisy_class_mean | 79824 | 85604 | 38161 | 308411 | 432176 |
| global_mean_centering__coefficient_0.25 | 410787 | 7797 | 10830 | 82586 | 101213 |
| global_mean_centering__coefficient_0.5 | 308156 | 13355 | 25894 | 164595 | 203844 |
| global_mean_centering__coefficient_1 | 183235 | 16834 | 45497 | 266434 | 328765 |
| gr_clip_style_two_sided__coefficient_1 | 347608 | 16712 | 19282 | 128398 | 164392 |

### P2 (Chowers orthogonal-complement candidates)

| candidate | orthogonality residual | recorded vs reconstructed v | normalizer mismatch | flips | exceptions |
|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 1.268e-08 | 8.876e-09 | 2.048e-07 | 4 | 0 |
| chowers_exact_projected_gap__coefficient_0.5 | 2.536e-08 | 9.888e-09 | 2.197e-07 | 4 | 0 |
| chowers_exact_projected_gap__coefficient_1 | 5.073e-08 | 3.184e-09 | 2.122e-07 | 10 | 0 |

## openai-clip-vit-b32-quickgelu__eurosat__sigma0.25 (300 items x 512 draws, 10 classes)

- bank archive `204e1dc5d493d6f6f53441aa24fa6b06b8bfcabda8108325aaf24497e2aec375`: all tensor hashes recompute = True; shard candidate metadata equals bank metadata = True
- recomputed no_correction vote vectors equal the stored count shard on 300/300 items; float64 re-scoring changes the base argmax on 2 of 153600 draws
- P5: rho = +0.9666, permutation p = 0.0000, hub class 1 (alignment rank 1), top-class share 0.4390
- P4a: MACE = 0.0006; bins (predicted -> observed): 0.000->0.000, 0.000->0.000, 0.003->0.003, 0.212->0.211, 0.659->0.659, 0.953->0.948, 1.000->1.000, 1.000->1.000, 1.000->1.000, 1.000->1.000
- P6: items already at k_min = 72; winner-runner-up margin quantiles q0.1=0.0003, q0.25=0.0008, q0.5=0.0019, q0.75=0.0038, q0.9=0.0068

### P1 containment and P7 flip budget

| candidate | applicable | flips | outside eligible | conforms | k0>=k_min | k0+E_F<k_min | undetermined | bound violations | E_F median | k1>=k_min |
|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | yes | 2 | 0 | yes | 72 | 228 | 0 | 0 | 0.0 | 72 |
| chowers_exact_projected_gap__coefficient_0.5 | yes | 2 | 0 | yes | 72 | 228 | 0 | 0 | 0.0 | 72 |
| chowers_exact_projected_gap__coefficient_1 | yes | 3 | 0 | yes | 72 | 228 | 0 | 0 | 0.0 | 72 |
| clean_boundary_active__step_0.0025 | yes | 677 | 0 | yes | 72 | 224 | 4 | 0 | 14.0 | 72 |
| clean_boundary_active__step_0.005 | yes | 1381 | 0 | yes | 72 | 217 | 11 | 0 | 27.0 | 74 |
| clean_boundary_active__step_0.01 | yes | 2648 | 0 | yes | 72 | 160 | 68 | 0 | 49.0 | 74 |
| clean_boundary_active__step_0.02 | yes | 5050 | 0 | yes | 72 | 62 | 166 | 0 | 125.5 | 74 |
| clean_boundary_active__step_0.04 | yes | 9429 | 0 | yes | 72 | 0 | 228 | 0 | 442.0 | 72 |
| cohen_aligned_noisy_margin__step_0.0025 | yes | 1929 | 0 | yes | 72 | 203 | 25 | 0 | 35.0 | 72 |
| cohen_aligned_noisy_margin__step_0.005 | yes | 3856 | 0 | yes | 72 | 125 | 103 | 0 | 70.0 | 71 |
| cohen_aligned_noisy_margin__step_0.01 | yes | 7865 | 0 | yes | 72 | 39 | 189 | 0 | 137.0 | 67 |
| cohen_aligned_noisy_margin__step_0.02 | yes | 15986 | 0 | yes | 72 | 0 | 228 | 0 | 284.5 | 66 |
| cohen_aligned_noisy_margin__step_0.04 | yes | 32076 | 0 | yes | 72 | 0 | 228 | 0 | 485.0 | 64 |
| control__learned_shared_translation_pure | yes | 44304 | 0 | yes | 72 | 0 | 228 | 0 | 512.0 | 68 |
| control__learned_shared_translation_tangent | yes | 75334 | 0 | yes | 72 | 0 | 228 | 0 | 512.0 | 54 |
| control__lowrank_tangent_r8 | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |
| control__noisy_class_mean | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |
| global_mean_centering__coefficient_0.25 | yes | 26176 | 0 | yes | 72 | 1 | 227 | 0 | 413.5 | 66 |
| global_mean_centering__coefficient_0.5 | yes | 98309 | 0 | yes | 72 | 0 | 228 | 0 | 512.0 | 51 |
| global_mean_centering__coefficient_1 | yes | 150274 | 0 | yes | 72 | 0 | 228 | 0 | 512.0 | 258 |
| gr_clip_style_two_sided__coefficient_1 | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |

### P9 winner-movement taxonomy (paired draws)

| candidate | unchanged | useful wrong->correct | harmful correct->wrong | wrong->wrong | total flips |
|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 153598 | 1 | 0 | 1 | 2 |
| chowers_exact_projected_gap__coefficient_0.5 | 153598 | 1 | 0 | 1 | 2 |
| chowers_exact_projected_gap__coefficient_1 | 153597 | 3 | 0 | 0 | 3 |
| clean_boundary_active__step_0.0025 | 152923 | 131 | 62 | 484 | 677 |
| clean_boundary_active__step_0.005 | 152219 | 243 | 119 | 1019 | 1381 |
| clean_boundary_active__step_0.01 | 150952 | 429 | 248 | 1971 | 2648 |
| clean_boundary_active__step_0.02 | 148550 | 806 | 460 | 3784 | 5050 |
| clean_boundary_active__step_0.04 | 144171 | 1595 | 824 | 7010 | 9429 |
| cohen_aligned_noisy_margin__step_0.0025 | 151671 | 147 | 337 | 1445 | 1929 |
| cohen_aligned_noisy_margin__step_0.005 | 149744 | 292 | 684 | 2880 | 3856 |
| cohen_aligned_noisy_margin__step_0.01 | 145735 | 551 | 1410 | 5904 | 7865 |
| cohen_aligned_noisy_margin__step_0.02 | 137614 | 1096 | 2857 | 12033 | 15986 |
| cohen_aligned_noisy_margin__step_0.04 | 121524 | 2350 | 5547 | 24179 | 32076 |
| control__learned_shared_translation_pure | 109296 | 3153 | 6496 | 34655 | 44304 |
| control__learned_shared_translation_tangent | 78266 | 13530 | 8871 | 52933 | 75334 |
| control__lowrank_tangent_r8 | 19320 | 80601 | 8452 | 45227 | 134280 |
| control__noisy_class_mean | 18865 | 67439 | 10588 | 56708 | 134735 |
| global_mean_centering__coefficient_0.25 | 127424 | 3116 | 3699 | 19361 | 26176 |
| global_mean_centering__coefficient_0.5 | 55291 | 11772 | 13374 | 73163 | 98309 |
| global_mean_centering__coefficient_1 | 3326 | 15115 | 17666 | 117493 | 150274 |
| gr_clip_style_two_sided__coefficient_1 | 78370 | 11464 | 8032 | 55734 | 75230 |

### P2 (Chowers orthogonal-complement candidates)

| candidate | orthogonality residual | recorded vs reconstructed v | normalizer mismatch | flips | exceptions |
|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 8.975e-09 | 1.018e-08 | 1.189e-07 | 2 | 0 |
| chowers_exact_projected_gap__coefficient_0.5 | 1.795e-08 | 6.855e-09 | 1.250e-07 | 2 | 0 |
| chowers_exact_projected_gap__coefficient_1 | 3.590e-08 | 5.046e-09 | 1.088e-07 | 3 | 0 |

## openai-clip-vit-l14-quickgelu__cifar100__sigma0.25 (1000 items x 512 draws, 100 classes)

- bank archive `c91814cfa9a0965daea63a746aa92a2bdcf4a861542ae71cf8151a278af013f8`: all tensor hashes recompute = True; shard candidate metadata equals bank metadata = True
- recomputed no_correction vote vectors equal the stored count shard on 998/1000 items; float64 re-scoring changes the base argmax on 2 of 512000 draws
- P5: rho = +0.6179, permutation p = 0.0000, hub class 65 (alignment rank 2), top-class share 0.0635
- P4a: MACE = 0.0023; bins (predicted -> observed): 0.000->0.000, 0.015->0.017, 0.230->0.221, 0.638->0.641, 0.917->0.923, 0.990->0.991, 1.000->0.999, 1.000->1.000, 1.000->1.000, 1.000->1.000
- P6: items already at k_min = 338; winner-runner-up margin quantiles q0.1=0.0010, q0.25=0.0027, q0.5=0.0072, q0.75=0.0183, q0.9=0.0360

### P1 containment and P7 flip budget

| candidate | applicable | flips | outside eligible | conforms | k0>=k_min | k0+E_F<k_min | undetermined | bound violations | E_F median | k1>=k_min |
|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | yes | 3 | 0 | yes | 338 | 662 | 0 | 0 | 0.0 | 338 |
| chowers_exact_projected_gap__coefficient_0.5 | yes | 3 | 0 | yes | 338 | 662 | 0 | 0 | 0.0 | 338 |
| chowers_exact_projected_gap__coefficient_1 | yes | 4 | 0 | yes | 338 | 662 | 0 | 0 | 0.0 | 338 |
| clean_boundary_active__step_0.0025 | yes | 1285 | 0 | yes | 338 | 651 | 11 | 0 | 5.0 | 339 |
| clean_boundary_active__step_0.005 | yes | 2574 | 0 | yes | 338 | 638 | 24 | 0 | 11.0 | 339 |
| clean_boundary_active__step_0.01 | yes | 5125 | 0 | yes | 338 | 593 | 69 | 0 | 21.0 | 341 |
| clean_boundary_active__step_0.02 | yes | 10238 | 0 | yes | 338 | 471 | 191 | 0 | 43.5 | 340 |
| clean_boundary_active__step_0.04 | yes | 19708 | 0 | yes | 338 | 259 | 403 | 0 | 94.0 | 341 |
| cohen_aligned_noisy_margin__step_0.0025 | yes | 484 | 0 | yes | 338 | 659 | 3 | 0 | 2.0 | 338 |
| cohen_aligned_noisy_margin__step_0.005 | yes | 952 | 0 | yes | 338 | 657 | 5 | 0 | 5.0 | 338 |
| cohen_aligned_noisy_margin__step_0.01 | yes | 1878 | 0 | yes | 338 | 646 | 16 | 0 | 10.0 | 338 |
| cohen_aligned_noisy_margin__step_0.02 | yes | 3761 | 0 | yes | 338 | 617 | 45 | 0 | 19.0 | 338 |
| cohen_aligned_noisy_margin__step_0.04 | yes | 7631 | 0 | yes | 338 | 553 | 109 | 0 | 39.0 | 338 |
| control__learned_shared_translation_pure | yes | 367401 | 0 | yes | 338 | 0 | 662 | 0 | 512.0 | 379 |
| control__learned_shared_translation_tangent | yes | 56566 | 0 | yes | 338 | 0 | 662 | 0 | 512.0 | 344 |
| control__lowrank_tangent_r8 | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |
| control__noisy_class_mean | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |
| global_mean_centering__coefficient_0.25 | yes | 69761 | 0 | yes | 338 | 21 | 641 | 0 | 357.0 | 333 |
| global_mean_centering__coefficient_0.5 | yes | 151546 | 0 | yes | 338 | 0 | 662 | 0 | 512.0 | 340 |
| global_mean_centering__coefficient_1 | yes | 247006 | 0 | yes | 338 | 0 | 662 | 0 | 512.0 | 347 |
| gr_clip_style_two_sided__coefficient_1 | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |

### P9 winner-movement taxonomy (paired draws)

| candidate | unchanged | useful wrong->correct | harmful correct->wrong | wrong->wrong | total flips |
|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 511997 | 1 | 0 | 2 | 3 |
| chowers_exact_projected_gap__coefficient_0.5 | 511997 | 0 | 0 | 3 | 3 |
| chowers_exact_projected_gap__coefficient_1 | 511996 | 0 | 0 | 4 | 4 |
| clean_boundary_active__step_0.0025 | 510715 | 164 | 117 | 1004 | 1285 |
| clean_boundary_active__step_0.005 | 509426 | 338 | 249 | 1987 | 2574 |
| clean_boundary_active__step_0.01 | 506875 | 664 | 503 | 3958 | 5125 |
| clean_boundary_active__step_0.02 | 501762 | 1256 | 1048 | 7934 | 10238 |
| clean_boundary_active__step_0.04 | 492292 | 2438 | 2214 | 15056 | 19708 |
| cohen_aligned_noisy_margin__step_0.0025 | 511516 | 52 | 82 | 350 | 484 |
| cohen_aligned_noisy_margin__step_0.005 | 511048 | 89 | 154 | 709 | 952 |
| cohen_aligned_noisy_margin__step_0.01 | 510122 | 186 | 308 | 1384 | 1878 |
| cohen_aligned_noisy_margin__step_0.02 | 508239 | 396 | 578 | 2787 | 3761 |
| cohen_aligned_noisy_margin__step_0.04 | 504369 | 799 | 1206 | 5626 | 7631 |
| control__learned_shared_translation_pure | 144599 | 24216 | 92556 | 250629 | 367401 |
| control__learned_shared_translation_tangent | 455434 | 7366 | 16303 | 32897 | 56566 |
| control__lowrank_tangent_r8 | 262419 | 63781 | 27094 | 158706 | 249581 |
| control__noisy_class_mean | 183344 | 76046 | 61928 | 190682 | 328656 |
| global_mean_centering__coefficient_0.25 | 442239 | 8078 | 12383 | 49300 | 69761 |
| global_mean_centering__coefficient_0.5 | 360454 | 15089 | 33006 | 103451 | 151546 |
| global_mean_centering__coefficient_1 | 264994 | 18190 | 63401 | 165415 | 247006 |
| gr_clip_style_two_sided__coefficient_1 | 334122 | 20363 | 30122 | 127393 | 177878 |

### P2 (Chowers orthogonal-complement candidates)

| candidate | orthogonality residual | recorded vs reconstructed v | normalizer mismatch | flips | exceptions |
|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 1.912e-08 | 3.416e-09 | 1.615e-07 | 3 | 0 |
| chowers_exact_projected_gap__coefficient_0.5 | 3.825e-08 | 1.703e-09 | 1.739e-07 | 3 | 0 |
| chowers_exact_projected_gap__coefficient_1 | 7.650e-08 | 1.379e-08 | 1.689e-07 | 4 | 0 |

## openai-clip-vit-l14-quickgelu__eurosat__sigma0.25 (300 items x 512 draws, 10 classes)

- bank archive `3196ac242872e746d0b76c07dc2a6b17b73bc020c82c8eca12afb4bf3213dd45`: all tensor hashes recompute = True; shard candidate metadata equals bank metadata = True
- recomputed no_correction vote vectors equal the stored count shard on 300/300 items; float64 re-scoring changes the base argmax on 0 of 153600 draws
- P5: rho = +0.7333, permutation p = 0.0219, hub class 0 (alignment rank 1), top-class share 0.8177
- P4a: MACE = 0.0008; bins (predicted -> observed): 0.000->0.000, 0.000->0.000, 0.002->0.003, 0.069->0.070, 0.420->0.420, 0.863->0.869, 0.991->0.991, 1.000->1.000, 1.000->1.000, 1.000->1.000
- P6: items already at k_min = 226; winner-runner-up margin quantiles q0.1=0.0021, q0.25=0.0051, q0.5=0.0095, q0.75=0.0135, q0.9=0.0179

### P1 containment and P7 flip budget

| candidate | applicable | flips | outside eligible | conforms | k0>=k_min | k0+E_F<k_min | undetermined | bound violations | E_F median | k1>=k_min |
|---|---|---|---|---|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | yes | 0 | 0 | yes | 226 | 74 | 0 | 0 | 0.0 | 226 |
| chowers_exact_projected_gap__coefficient_0.5 | yes | 0 | 0 | yes | 226 | 74 | 0 | 0 | 0.0 | 226 |
| chowers_exact_projected_gap__coefficient_1 | yes | 0 | 0 | yes | 226 | 74 | 0 | 0 | 0.0 | 226 |
| clean_boundary_active__step_0.0025 | yes | 457 | 0 | yes | 226 | 59 | 15 | 0 | 1.0 | 227 |
| clean_boundary_active__step_0.005 | yes | 935 | 0 | yes | 226 | 46 | 28 | 0 | 3.0 | 225 |
| clean_boundary_active__step_0.01 | yes | 1881 | 0 | yes | 226 | 13 | 61 | 0 | 7.0 | 224 |
| clean_boundary_active__step_0.02 | yes | 3801 | 0 | yes | 226 | 1 | 73 | 0 | 27.0 | 222 |
| clean_boundary_active__step_0.04 | yes | 8016 | 0 | yes | 226 | 0 | 74 | 0 | 167.5 | 214 |
| cohen_aligned_noisy_margin__step_0.0025 | yes | 599 | 0 | yes | 226 | 56 | 18 | 0 | 1.0 | 225 |
| cohen_aligned_noisy_margin__step_0.005 | yes | 1251 | 0 | yes | 226 | 30 | 44 | 0 | 4.0 | 224 |
| cohen_aligned_noisy_margin__step_0.01 | yes | 2464 | 0 | yes | 226 | 7 | 67 | 0 | 10.0 | 221 |
| cohen_aligned_noisy_margin__step_0.02 | yes | 5025 | 0 | yes | 226 | 2 | 72 | 0 | 45.0 | 216 |
| cohen_aligned_noisy_margin__step_0.04 | yes | 10589 | 0 | yes | 226 | 0 | 74 | 0 | 410.5 | 209 |
| control__learned_shared_translation_pure | yes | 14591 | 0 | yes | 226 | 0 | 74 | 0 | 512.0 | 259 |
| control__learned_shared_translation_tangent | yes | 108178 | 0 | yes | 226 | 0 | 74 | 0 | 512.0 | 92 |
| control__lowrank_tangent_r8 | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |
| control__noisy_class_mean | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |
| global_mean_centering__coefficient_0.25 | yes | 6400 | 0 | yes | 226 | 1 | 73 | 0 | 18.5 | 223 |
| global_mean_centering__coefficient_0.5 | yes | 32520 | 0 | yes | 226 | 0 | 74 | 0 | 484.5 | 194 |
| global_mean_centering__coefficient_1 | yes | 115881 | 0 | yes | 226 | 0 | 74 | 0 | 512.0 | 216 |
| gr_clip_style_two_sided__coefficient_1 | no: containment applies only to pure/shared-tangent prototype operators with identity image transform |  |  |  |  |  |  |  |  |  |

### P9 winner-movement taxonomy (paired draws)

| candidate | unchanged | useful wrong->correct | harmful correct->wrong | wrong->wrong | total flips |
|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 153600 | 0 | 0 | 0 | 0 |
| chowers_exact_projected_gap__coefficient_0.5 | 153600 | 0 | 0 | 0 | 0 |
| chowers_exact_projected_gap__coefficient_1 | 153600 | 0 | 0 | 0 | 0 |
| clean_boundary_active__step_0.0025 | 153143 | 66 | 99 | 292 | 457 |
| clean_boundary_active__step_0.005 | 152665 | 133 | 206 | 596 | 935 |
| clean_boundary_active__step_0.01 | 151719 | 253 | 433 | 1195 | 1881 |
| clean_boundary_active__step_0.02 | 149799 | 495 | 880 | 2426 | 3801 |
| clean_boundary_active__step_0.04 | 145584 | 1017 | 1882 | 5117 | 8016 |
| cohen_aligned_noisy_margin__step_0.0025 | 153001 | 130 | 92 | 377 | 599 |
| cohen_aligned_noisy_margin__step_0.005 | 152349 | 263 | 179 | 809 | 1251 |
| cohen_aligned_noisy_margin__step_0.01 | 151136 | 482 | 388 | 1594 | 2464 |
| cohen_aligned_noisy_margin__step_0.02 | 148575 | 960 | 826 | 3239 | 5025 |
| cohen_aligned_noisy_margin__step_0.04 | 143011 | 1977 | 1892 | 6720 | 10589 |
| control__learned_shared_translation_pure | 139009 | 1992 | 2377 | 10222 | 14591 |
| control__learned_shared_translation_tangent | 45422 | 22759 | 10496 | 74923 | 108178 |
| control__lowrank_tangent_r8 | 16815 | 82955 | 4849 | 48981 | 136785 |
| control__noisy_class_mean | 15883 | 71254 | 6642 | 59821 | 137717 |
| global_mean_centering__coefficient_0.25 | 147200 | 1126 | 2381 | 2893 | 6400 |
| global_mean_centering__coefficient_0.5 | 121080 | 5258 | 8952 | 18310 | 32520 |
| global_mean_centering__coefficient_1 | 37719 | 12383 | 14882 | 88616 | 115881 |
| gr_clip_style_two_sided__coefficient_1 | 99069 | 15054 | 6582 | 32895 | 54531 |

### P2 (Chowers orthogonal-complement candidates)

| candidate | orthogonality residual | recorded vs reconstructed v | normalizer mismatch | flips | exceptions |
|---|---|---|---|---|---|
| chowers_exact_projected_gap__coefficient_0.25 | 5.188e-09 | 7.049e-09 | 1.140e-07 | 0 | 0 |
| chowers_exact_projected_gap__coefficient_0.5 | 1.038e-08 | 3.890e-09 | 1.167e-07 | 0 | 0 |
| chowers_exact_projected_gap__coefficient_1 | 2.075e-08 | 9.416e-09 | 9.484e-08 | 0 | 0 |

## Comparison with the registered d1_08 output

`results/satml2027_local/analysis/N3C-20260920-V3/n3c_predictions.json`: 1793 numeric leaves compared, max abs deviation 0.000e+00, 0 mismatches, 0 leaves absent from this reconstruction; k_min match = True

