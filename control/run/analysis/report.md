# Phase-4 FARLA result report: EXP-20260917-020-FARLA-FULL

Decision: **CONTINUE_PAPER_CANDIDATE**

| Candidate | Clean | Standard CA | Anchored CA | Certified wrong | Clean gate |
|---|---:|---:|---:|---:|:---:|
| no_correction | 0.6088 | 0.1160 | 0.0843 | 0.3417 | True |
| shared_translation_ce_supervised | 0.6128 | 0.1175 | 0.0868 | 0.3342 | True |
| lowrank_ce_r8_supervised | 0.7250 | 0.2710 | 0.2335 | 0.1358 | True |
| lowrank_tail_r8_supervised | 0.7080 | 0.3650 | 0.2875 | 0.0905 | True |
| farla_full_r4_supervised | 0.6215 | 0.3335 | 0.2460 | 0.1177 | True |
| farla_full_r8_supervised | 0.6858 | 0.3505 | 0.2955 | 0.1042 | True |
| farla_full_direct_supervised | 0.6623 | 0.3460 | 0.3120 | 0.1143 | False |
| farla_full_r8_pseudo | 0.4698 | 0.2710 | 0.2115 | 0.1775 | False |

## Predeclared paired contrasts

| Contrast | Proposed - comparator | 95% CI |
|---|---:|---:|
| farla_vs_no_correction | 0.2345 | [0.2025, 0.2662] |
| farla_vs_lowrank_ce | 0.0795 | [0.0473, 0.1115] |
| farla_vs_lowrank_tail | -0.0145 | [-0.0405, 0.0112] |
| lowrank_ce_vs_shared_translation | 0.1535 | [0.1270, 0.1803] |
| farla_r8_vs_r4 | 0.0170 | [-0.0065, 0.0400] |

- clean_constraint=True
- positive_macro_gain=True
- paired_interval_above_zero:farla_vs_no_correction=True
- paired_interval_above_zero:farla_vs_lowrank_ce=True
