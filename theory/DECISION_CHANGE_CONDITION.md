# Decision-Change Conditions for Shared Prototype Operators

**Status:** primary derivation complete and numerically verified (27/27 new tests pass); independent second-member check pending  
**Claims touched:** the registered SaTML abstract's sentence "exact per-draw identities and necessary decision-change conditions for the studied shared prototype operators"; complements C1 (clean-gap non-identification); documents the mathematics behind the registered N3C predictions P1, P2 and P7 in `configs/satml2027/n3c_v3.json`  
**Engine boundary:** everything below is geometry of the hard-label base classifier. It contains no certificate, no FSS quantity (`D*`, `tau*`, FCSB), and no Cohen probability. Cohen certificates are computed separately from hard-label counts and are never reconstructed from anything in this note.  
**Paper blocks:** `paper/satml2027/sections/theory_decision_change.tex` (statements, labels `prop:identity`, `prop:flip`, `cor:chowers`, `cor:budget`, `rem:twosided`) and `paper/satml2027/sections/appendix_proofs.tex` (`app:proofs`)  
**Tests:** `tests/test_decision_change_condition.py`, `tests/test_negative_paper_gate_n3_encoder_monte_carlo.py`  
**Last updated:** 2026-09-25

This note uses plain-text mathematics so that it is readable without a Markdown
math engine. `‖·‖` is the Euclidean (ℓ2) norm and `⟨a, b⟩ = aᵀb`.

## 0. Purpose and scope

The registered abstract promises exact per-draw identities and necessary
decision-change conditions for the shared prototype operators that the audit
tested. Until now those statements existed only as code docstrings
(`satml2027/common/banks.py:296-301`, `satml2027/day1/d1_08_n3c_predictions.py:6-22`)
and as the registered prediction P1 in `configs/satml2027/n3c_v3.json`. This
note states and proves them, confirms the exact operator forms from the code,
records how the statements are used by the registered N3C analysis, and lists
the numerical verification.

What the note establishes:

- Proposition 1: an exact per-draw identity for every renormalized common text
  translation.
- Proposition 2: a necessary condition for any pairwise or top-1 decision
  change, with a sharp bound in terms of the per-class normalizers and the
  translation norm.
- Corollary 1: the Chowers-style exact projected-gap translation never changes
  a decision, at any coefficient.
- Corollary 2: a finite flip budget on a fixed set of draws (the registered
  N3C P7 form).
- Two remarks: why the two-sided GR-CLIP-style operator escapes the bound, and
  how the statements relate to the Gate-N3 non-identification construction.

What the note does not establish: any population probability, any certified
radius, any efficacy or inefficacy claim about a real model, or anything about
class-conditioned, image-side, prompt-side, or low-rank operators.

## 1. Setting

```text
z ∈ ℝ^d, ‖z‖ = 1          one image feature: the clean feature or one normalized noisy draw
t_1, …, t_K, ‖t_k‖ = 1     frozen unit text prototypes
s_k = ⟨z, t_k⟩             original cosine scores
decision                  argmax_k s_k   (hard label; a softmax temperature never changes an argmax)
```

A common text translation adds one vector `w` to every prototype and
renormalizes:

```text
t'_k = (t_k + w) / n_k,    n_k = ‖t_k + w‖ > 0
c    = ⟨z, w⟩
```

The bank builders refuse any raw row with norm at or below `1e-12`
(`satml2027/common/banks.py:116-121`, `interventions/boundary_active.py:308-311`),
so `n_k > 0` holds for every registered bank.

## 2. The operators as implemented

The exact forms were confirmed from the frozen EXP-017 rule
(`vlm_pipeline/phase3_candidates.py`, `interventions/core.py`,
`interventions/boundary_active.py`) and its line-by-line port used by the
registered extension (`satml2027/common/banks.py::build_eighteen_candidate_banks`).
`μ_T` is the uniform prototype mean (`banks.py:166-168`; `core.py:429-435` uses
uniform weights by default) and `μ_I` the mean of the development clean unit
image features (`banks.py:165`; `phase3_candidates.py:166`).

| Candidate family (grid) | Code | Text side `w` | Image side |
|---|---|---|---|
| `no_correction` | `banks.py:190-191` | `w = 0` | identity |
| `global_mean_centering__coefficient_c`, `c ∈ {0.25, 0.5, 1}` | `banks.py:197-201`; `core.py:501-507`; `phase3_candidates.py:187` | `w = −c (μ_T − μ_I) = c (μ_I − μ_T)` | identity |
| `chowers_exact_projected_gap__coefficient_c`, `c ∈ {0.25, 0.5, 1}` | `banks.py:218-227`; `core.py:853-871, 917-924`; `phase3_candidates.py:220` | `w = c p`, `p = (I − VVᵀ)(μ_I − μ_T)`, `V` = orthonormal basis of `span{t_k − μ_T}` from a thresholded SVD | identity |
| `clean_boundary_active__step_α`, `cohen_aligned_noisy_margin__step_α`, `α ∈ {0.0025, 0.005, 0.01, 0.02, 0.04}` | `banks.py:236-239`; `boundary_active.py:290-314`; `phase3_candidates.py:241` | `w = α u`, `u` a unit development direction (`‖u‖ = 1` validated with `atol = 2e-5`) | identity |
| `gr_clip_style_two_sided__coefficient_1` | `banks.py:208-215`; `core.py:697-706`; `phase3_candidates.py:40-58` | `w = −μ_T` (coefficient fixed at 1) | `z ↦ (z − μ_I)/‖z − μ_I‖` |

Registered controls (outside the frozen grid, `banks.py:254-293`):

| Control | Form | Covered by |
|---|---|---|
| `control__learned_shared_translation_pure` | `t'_k = (t_k + v)/n_k`, learned `v` | Propositions 1-2 with `w = v` |
| `control__learned_shared_translation_tangent` | `t'_k = normalize(t_k + v − ⟨v, t_k⟩ t_k) = (a_k t_k + v)/n_k`, `a_k = 1 − ⟨v, t_k⟩` | Section 7 (per-class scales) |
| `control__noisy_class_mean`, `control__lowrank_tangent_r8` | class-specific rows; not a common translation | not covered (P1/P7 mark them not applicable, `d1_08_n3c_predictions.py:178-183`) |

Two corrections to the working assumptions in the task brief:

1. Text mean-centering is **not** `normalize(t_k − c μ_T)`. The code subtracts
   the scaled *gap* `c (μ_T − μ_I)` (`banks.py:197-201`; `core.py:501-503`;
   the manifest string at `core.py:468` reads
   `normalize(t_k - coefficient * (mu_text - mu_image))`). The paper's operator
   list in `paper/satml2027/sections/operators.tex` states this correctly.
2. The GR-CLIP-style operator has no coefficient parameter (`GRCLIPStyle`
   takes only the two means); the registered grid uses coefficient 1 on both
   sides. Section 8 states the identity for coefficient 1 as implemented and
   the general-`c` form as a remark.

The registered extension records the exact operator parameters
`(a_k, n_k, v)` at bank-construction time (`banks.py:296-326`), so the N3C
analysis evaluates P1 and P7 with exact normalizers rather than recovered ones.

## 3. Proposition 1 (per-draw identity)

> **Proposition 1.** For every unit `z` and every `w` with all `n_k > 0`,
>
> ```text
> score'_k := ⟨z, t'_k⟩ = (s_k + c) / n_k .
> ```
>
> Hence, for every pair `(j, k)`,
>
> ```text
> score'_j − score'_k = [ n_k (s_j + c) − n_j (s_k + c) ] / (n_j n_k)        (1)
>
> n_j² − n_k² = 2 ⟨t_j − t_k, w⟩,   so   n_j − n_k = 2 ⟨t_j − t_k, w⟩ / (n_j + n_k).   (2)
> ```

*Proof.* Linearity gives `⟨z, t_k + w⟩ / n_k = (s_k + c)/n_k`. Subtracting the
expressions for `j` and `k` over the common denominator `n_j n_k` gives (1).
For (2), `n_k² = ‖t_k‖² + 2⟨t_k, w⟩ + ‖w‖² = 1 + 2⟨t_k, w⟩ + ‖w‖²`, so
`n_j² − n_k² = 2⟨t_j − t_k, w⟩`; dividing by `n_j + n_k > 0` gives the second
form. ∎

Consequences used below: the translation enters every score through the same
scalar `c`, and the only class-dependent effect of the translation is the
per-class normalizer `n_k`. In particular, if all `n_k` were equal, the ranking
of the scores could not change at all.

## 4. Proposition 2 (necessary decision-change condition)

> **Proposition 2.** Let `s_j ≥ s_k` (class `j` wins the pair originally) and
> suppose that after the translation `score'_k ≥ score'_j` (class `k` scores at
> least `j`). Then
>
> ```text
> 0 ≤ s_j − s_k ≤ (n_j − n_k) · score'_j ≤ |n_j − n_k|
>              = 2 |⟨t_j − t_k, w⟩| / (n_j + n_k) ≤ min{1, ‖w‖} · ‖t_j − t_k‖ .      (3)
> ```
>
> In particular:
>
> - (small-step form) for `‖w‖ = α`, every pair whose order changes has original
>   cosine margin at most `α ‖t_j − t_k‖ ≤ 2α`;
> - (top-1 form) if the top-1 label of `z` changes from `j` to `k`, then the
>   original top-1 margin of `z` (winner minus runner-up) is at most
>   `|n_j − n_k| ≤ max_{i,l} |n_i − n_l|`, and hence at most
>   `min{1, ‖w‖} · max_{i,l} ‖t_i − t_l‖ ≤ 2 ‖w‖`;
> - (direction) if `score'_j > 0` and `s_j > s_k`, then `n_j > n_k`: the original
>   winner can lose only to a class whose translated raw norm is strictly
>   smaller, i.e. only when `⟨t_j − t_k, w⟩ > 0`.

*Proof.*

**Step 1 (rearrangement).** By Proposition 1 the hypothesis reads
`(s_k + c)/n_k ≥ (s_j + c)/n_j`. Multiply by `n_k > 0`:
`s_k + c ≥ (n_k / n_j)(s_j + c)`. Subtract `s_j + c` from both sides:

```text
s_k − s_j ≥ (n_k/n_j − 1)(s_j + c) = (n_k − n_j) (s_j + c)/n_j = (n_k − n_j) · score'_j ,
```

that is, `s_j − s_k ≤ (n_j − n_k) · score'_j`. Multiplying the hypothesis by
`n_j` instead gives the companion bound `s_j − s_k ≤ (n_j − n_k) · score'_k`.
The lower bound `0 ≤ s_j − s_k` is the assumption `s_j ≥ s_k`.

**Step 2 (Cauchy-Schwarz).** `z` and `t'_j` are unit vectors, so
`|score'_j| ≤ 1` and `(n_j − n_k) · score'_j ≤ |n_j − n_k|`.

**Step 3 (exact form).** By (2), `|n_j − n_k| = 2|⟨t_j − t_k, w⟩| / (n_j + n_k)`.

**Step 4 (reverse triangle inequality).**
`|n_j − n_k| = | ‖t_j + w‖ − ‖t_k + w‖ | ≤ ‖t_j − t_k‖`.

**Step 5 (Ptolemy's inequality).** Claim: `|n_j − n_k| ≤ ‖w‖ · ‖t_j − t_k‖`.
For `w = 0` both sides vanish. For `w ≠ 0`, use the inversion
`x ↦ x* = x / ‖x‖²` of nonzero vectors, which satisfies

```text
‖x* − y*‖ = ‖x − y‖ / (‖x‖ ‖y‖)
```

(expand both squared norms: `1/‖x‖² − 2⟨x,y⟩/(‖x‖²‖y‖²) + 1/‖y‖²
= (‖y‖² − 2⟨x,y⟩ + ‖x‖²)/(‖x‖²‖y‖²)`). Apply the ordinary triangle
inequality to the images of the three nonzero points `t_j`, `t_k`, `−w`;
since `t_j* = t_j` and `t_k* = t_k`,

```text
‖t_j + w‖ / ‖w‖ = ‖t_j* − (−w)*‖ ≤ ‖t_j* − t_k*‖ + ‖t_k* − (−w)*‖ = ‖t_j − t_k‖ + ‖t_k + w‖ / ‖w‖ .
```

Multiplying by `‖w‖` gives `n_j − n_k ≤ ‖w‖ ‖t_j − t_k‖`; exchanging `j` and
`k` gives the absolute value. (This is Ptolemy's inequality for the four
points `t_j, t_k, −w, 0`.) Steps 4 and 5 together give the factor
`min{1, ‖w‖}`.

Two remarks on Step 5.

- The route suggested by the exact form, Cauchy-Schwarz on
  `2|⟨t_j − t_k, w⟩| / (n_j + n_k) ≤ 2‖w‖‖t_j − t_k‖/(n_j + n_k)`, proves the
  claim only when `n_j + n_k ≥ 2`. That condition fails for translations such
  as `w = −β t_k` with `β` near 1 (then `n_k = |1 − β|` is small), so the
  inversion argument is needed for the general statement. The test
  `test_normalizer_difference_lemma_including_adversarial_translations`
  exercises exactly such translations.
- The claim is sharp. With unit `t_j, t_k`, `‖t_j − t_k‖ = 6/5`, and
  `w = −t_k + (7/20) (t_j + t_k)/‖t_j + t_k‖`, one gets `n_k = 7/20`,
  `n_j = 5/4`, `‖w‖ = 3/4`, and both sides equal `9/10`. Geometrically this is
  the equality case of the triangle inequality: the inverted point `(−w)*`
  lies on the line through `t_j` and `t_k`, beyond `t_k`. So the factor `‖w‖`
  cannot be improved without further assumptions; for small `‖w‖` the bound is
  attained to first order whenever `w` is parallel to `t_j − t_k`.

**Step 6 (consequences).** With `‖w‖ = α` and `‖t_j − t_k‖ ≤ 2` (unit vectors)
the chain gives `s_j − s_k ≤ α ‖t_j − t_k‖ ≤ 2α`. If the top-1 label changes
from `j` to `k`, then `s_j ≥ s_i` for all `i` and `score'_k ≥ score'_i` for all
`i`, so the pair `(j, k)` satisfies the hypotheses; the original top-1 margin
obeys `s_j − max_{i≠j} s_i ≤ s_j − s_k ≤ |n_j − n_k| ≤ max_{i,l} |n_i − n_l|`.
Finally, if `score'_j > 0` and `s_j > s_k`, Step 1 forces `n_j − n_k > 0`. ∎

Reading of Proposition 2 for the registered grid: a step `α` can change the
hard label of a noisy draw only if that draw's original winner-runner-up
margin is at most `α · max_{i,l} ‖t_i − t_l‖ ≤ 2α`. This is a property of the
saved original scores alone; it can be evaluated before any candidate bank is
scored.

## 5. Corollary 1 (Chowers projection is decision-invariant)

> **Corollary 1.** If `⟨w, t_i − t_j⟩ = 0` for all `i, j`, then all `n_k` equal
> one value `n`, and `score'_k = (s_k + c)/n` with `n` and `c` independent of
> `k`. The argmax (including the set of tied maximizers) is unchanged for every
> `z`. On any fixed set of noisy draws scored against both banks, the operator
> therefore produces identical hard labels, vote counts, selected classes,
> one-sided confidence bounds, certified radii, and accuracies: it is an exact
> invariance (negative) control, not a competing method. This applies to the
> exact projected-gap translation `w = c p` at every coefficient `c`.

*Proof.* For the projected-gap translation, `p = (I − VVᵀ)(μ_I − μ_T)` with `V`
an orthonormal basis of `span{t_k − μ_T}` (`banks.py:218-223`;
`core.py:853-871`). Since `t_i − t_j = (t_i − μ_T) − (t_j − μ_T)` lies in that
span, `⟨p, t_i − t_j⟩ = 0` for all `i, j`, and every multiple `w = c p`
inherits this. Then `⟨t_k, w⟩ = ⟨t_k − t_1, w⟩ + ⟨t_1, w⟩ = ⟨t_1, w⟩` for every
`k`, so `n_k² = 1 + 2⟨t_1, w⟩ + ‖w‖²` is one number `n²` for all classes. By
Proposition 1, `score'_k = (s_k + c)/n` is a strictly increasing affine
function of `s_k` with coefficients independent of `k`, so the maximizer set is
unchanged for every `z`. Identical hard labels on a fixed draw set imply
identical vote counts, and every downstream quantity of hard-label smoothing
(selected class, Clopper-Pearson bound, radius, standard and anchored
accuracy, abstention) is a function of those counts. ∎

Observed match with the saved EXP-017 outcome. In
`artifacts/negative_paper/exp017_scientific_soundness_audit_v1.json`, the
equal-cell-weight aggregate blocks of `chowers_exact_projected_gap__coefficient_0.25`,
`__coefficient_0.5`, and `__coefficient_1` are identical to one another and to
the `no_correction` block (all metrics at all radii, abstention, anchor
agreement, clean and smoothed accuracy, and censoring counts). This was checked
on 2026-09-25 by a programmatic JSON comparison of the four blocks; no value
is transcribed here. The paper's generated macro `\ChowersMaxAbsDelta`
(`paper/satml2027/MACROS_CONTRACT.md`) is the corresponding generated
quantity. The identity of the outcomes is exactly what Corollary 1 predicts
when every candidate bank is scored on the same noise draws, which is how the
workers operate (`satml2027/day1/d1_05_run_shard.py:125-133` scores all
identity-transform banks on one feature batch).

Floating-point caveat. Corollary 1 is exact in real arithmetic. In float32
scoring, a draw whose original margin is at the level of roundoff may change
its label; such float-level flips are numerical ties, they still satisfy
Proposition 2 (the identity holds up to roundoff and the registered analysis
applies a `1e-6` tie tolerance, `d1_08_n3c_predictions.py:48,198`), and the
registered prediction P2 reports them explicitly. The test
`test_orthogonal_projection_float32_flips_are_confined_to_numerical_ties`
checks that any float32 flip has an exact original margin below `1e-5`.

## 6. Corollary 2 (finite flip budget)

> **Corollary 2.** Fix `N` draws `z_1, …, z_N` (for example the fresh noisy
> features of one item). Let `c_i` be the number of draws whose original argmax
> is `i`, `k_0 = max_i c_i`, and let
>
> ```text
> E_F = #{ m : the original winner j of z_m has some k ≠ j with s_j − s_k ≤ |n_j − n_k| } .
> ```
>
> Then `E_F ≤ #{ m : original top-1 margin of z_m ≤ max_{i,l} |n_i − n_l| }`,
> and after the translation the counts `c'_i` satisfy
>
> ```text
> c_i − E_F ≤ c'_i ≤ c_i + E_F   for every i,   hence   max_i c'_i ≤ k_0 + E_F .
> ```
>
> In particular, if `k_0 + E_F < k_min` for a confirmation threshold `k_min`,
> then no class reaches `k_min` on these draws after the translation.

*Proof.* Let `F` be the set of draws whose argmax changes and, for `m ∈ F`,
let `j_m` and `k_m ≠ j_m` be the original and translated winners. Then
`s_{j_m} ≥ s_{k_m}` and `score'_{k_m} ≥ score'_{j_m}`, so Proposition 2 gives
`s_{j_m} − s_{k_m} ≤ |n_{j_m} − n_{k_m}|`: every `m ∈ F` is eligible and
`|F| ≤ E_F`. A draw eligible through the pair `(j, k)` has top-1 margin
`s_j − max_{i≠j} s_i ≤ s_j − s_k ≤ |n_j − n_k| ≤ max_{i,l} |n_i − n_l|`, which
gives the coarse bound on `E_F`. Class `i` loses the votes of the draws in `F`
with `j_m = i` and gains those with `k_m = i`, so
`c'_i ≤ c_i + #{m ∈ F : k_m = i} ≤ c_i + E_F ≤ k_0 + E_F` and
`c'_i ≥ c_i − E_F`. If `k_0 + E_F < k_min` then `max_i c'_i < k_min`. ∎

How the registered analysis uses it (`d1_08_n3c_predictions.py:190-225`,
prediction P7 in `configs/satml2027/n3c_v3.json`). For every item, candidate
and cell, the analysis computes from the *original* scores the per-draw
winner-competitor margins, the exact normalizers `n_k` recorded at bank
construction, and hence `E_F` (the code's `budget`), and reports the partition

```text
{ k_0 ≥ k_min }              the uncorrected count already reaches the threshold
{ k_0 + E_F < k_min }        the threshold is unattainable on these draws for this candidate
{ undetermined }             otherwise
```

together with `bound_violations = #{ items : max_i c'_i > k_0 + E_F }`, which
Corollary 2 says must be zero. Because `E_F` needs only saved original margins
and the operator's normalizers, one can predict from saved data, without
rescoring against the candidate bank, whether a translation step could move a
confirmation count across `k_min`. Three cautions:

- `k_min` is the registered confirmation count at which the exact one-sided
  lower confidence bound reaches the target radius
  (`satml2027/common/certify.py:34-47`); the registration stores it for the
  registered draw budget. The partition is a statement about a fixed set of
  draws, not about the population probability.
- Cohen certification selects a class on separate draws and confirms it on
  others. Corollary 2 applies to whichever fixed draw set is scored against
  both banks (paired noise); the bound `max_i c'_i ≤ k_0 + E_F` covers every
  class, so it covers the translated bank's own selected class.
- Following the registered stop rule, the phrase "cannot be certified" is not
  used; the correct wording is "the threshold is unattainable on these draws".
  Nothing in the partition is a certificate.

## 7. Extension: per-class scales (tangent controls)

The registered tangent controls have the form `t'_k = (a_k t_k + v)/n_k` with
`a_k = 1 − ⟨v, t_k⟩` and `n_k = ‖a_k t_k + v‖` (`banks.py:271-277`); the
registered condition requires every `a_k > 0` (`d1_08_n3c_predictions.py:190-194`).
Repeating Step 1 of Proposition 2 with `a_k s_k` in place of `s_k` gives, for a
pair whose order changes,

```text
a_j s_j − a_k s_k ≤ (n_j − n_k) · score'_j ≤ |n_j − n_k| ,
```

and therefore, since `|s_k| ≤ 1`,

```text
a_j (s_j − s_k) = a_j s_j − a_k s_k + (a_k − a_j) s_k ≤ |n_j − n_k| + |a_j − a_k| .
```

This is the per-class-scale form of P1 in the docstring
(`d1_08_n3c_predictions.py:8-11`) and in the code
(`threshold = |n_k − n_j| + |a_k − a_j|`, `eligible = a_winner · margin ≤ threshold + tol`,
lines 197-198). For a pure translation `a_k = 1` and it reduces to
Proposition 2.

## 8. Remark: two-sided operators

The GR-CLIP-style operator also transforms the image feature,
`z ↦ z' = (z − μ_I)/‖z − μ_I‖` (`phase3_candidates.py:40-58`;
`core.py:698-706`; scored as such by the worker, `d1_05_run_shard.py:128-133`).
Renormalizing `z` never changes an argmax, but the shift does. With
`n_k = ‖t_k − μ_T‖`,

```text
score'_k = ⟨z − μ_I, t_k − μ_T⟩ / (‖z − μ_I‖ n_k)
         ∝ [ s_k − ⟨z, μ_T⟩ − ⟨μ_I, t_k⟩ + ⟨μ_I, μ_T⟩ ] / n_k ,
```

where the proportionality constant `1/‖z − μ_I‖` is common to all classes. The
first two numerator terms are the text-only identity of Proposition 1 with
`w = −μ_T`, and the last term is common to all classes; the third term
`−⟨μ_I, t_k⟩` is a per-class bias that no common text translation produces.
The argument of Proposition 2 therefore does not apply, and a two-sided
operator can change the label of a draw whose original margin exceeds every
text-only bound in (3). An explicit instance: prototypes `e_1, e_2, e_3`,
`z = normalize(e_1 + 0.05 e_2)`, `μ_T` the prototype mean, `μ_I = 0.98 e_1`.
The bank is symmetric, so all `n_k` are equal and the exact text-only bound
`|n_j − n_k|` is zero, the coarse bound `‖w‖ ‖t_1 − t_2‖` is below the original
margin, the text-only translation `w = −μ_T` leaves the label at class 1, and
the two-sided operator moves it to class 2
(`test_two_sided_centering_can_flip_beyond_every_text_only_bound`).

For a hypothetical two-sided operator with a common coefficient `c` on both
sides, `z ↦ normalize(z − c μ_I)`, `t'_k = (t_k − c μ_T)/n_k`, the same
expansion gives `score'_k ∝ [ s_k − c⟨z, μ_T⟩ − c⟨μ_I, t_k⟩ + c²⟨μ_I, μ_T⟩ ] / n_k`.
Only `c = 1` is registered.

This is why the two-sided operator is the only registered operator that can
act where the text-only translations cannot, and why its outcome is not
bounded by Proposition 2. It is also why the registered N3C analysis marks
P1/P7 as not applicable for it (`d1_08_n3c_predictions.py:178-183`).

## 9. Remark: relation to non-identification (Gate N3)

The binary Cohen construction (`research/NEGATIVE_PAPER_GATE_N3_COHEN_NONIDENTIFICATION.md`,
`configs/negative_paper_gate_n3_theory_v1.json`) shows that the clean global
gap, the clean feature, the clean margin, and the prototype correlation do not
determine the population Cohen radius: two banks share all of them and have
radii `q` and `q/κ`. Propositions 1-2 are the complementary quantitative
statement for the operators actually tested: whatever the gap, a text-side
common translation can change the hard label only of draws whose original
margin lies inside the window

```text
max_{i,l} |n_i − n_l| ≤ min{1, ‖w‖} · max_{i,l} ‖t_i − t_l‖ ≤ 2 ‖w‖ .
```

The non-identification construction says the clean summary is silent about
the noisy response; Proposition 2 says what a small shared translation can do
to the noisy response once the noisy draws are in hand. Neither statement is
a certificate. Cohen certificates are computed by the separate hard-label
procedure from vote counts, and nothing in this note is substituted for a
count, a lower confidence bound, or a radius.

## 10. Relation to the registered docstring inequality

The registered P1 statement (`d1_08_n3c_predictions.py:6-11`;
`configs/satml2027/n3c_v3.json`, prediction `P1`) reads: for every draw whose
argmax changes from `j` to `k`, `|⟨z, t_j − t_k⟩| ≤ |n_k − n_j|`, with the
per-class-scale form for tangent controls and tie tolerance `1e-6`. Comparison
with Proposition 2:

1. **No contradiction.** The docstring inequality is the Step-2 relaxation of
   (3): for the (original top-1, new top-1) pair, `s_j − s_k ≥ 0`, so
   `|⟨z, t_j − t_k⟩| = s_j − s_k ≤ |n_j − n_k|`. The registered prediction
   remains valid as written, and `banks.py:296-301` correctly notes that the
   condition needs only differences of the recorded `n_k` and `a_k`.
2. **The exact statement is strictly sharper.** Proposition 2 gives
   `s_j − s_k ≤ (n_j − n_k) · score'_j` (and the companion bound with
   `score'_k`). The docstring's absolute values discard (a) the factor
   `score'_j`, the translated cosine score of the original winner, which is
   below 1 in absolute value, and (b) the direction: when `score'_j > 0`, a
   flip requires `n_j > n_k`, i.e. `⟨t_j − t_k, w⟩ > 0`. A draw whose original
   winner would be *shortened* relative to the challenger by the translation
   cannot flip while its translated score is positive, even if its margin is
   inside the absolute window.
3. **Quantifier.** The docstring quantifies over the old and new top-1
   classes; Proposition 2 holds for every pair whose order changes, and the
   top-1 statement is the special case used by P1.
4. **The last inequality in (3) is not a Cauchy-Schwarz consequence of the
   identity.** `|n_j − n_k| ≤ ‖w‖ ‖t_j − t_k‖` requires the inversion (Ptolemy)
   argument of Step 5 because `n_j + n_k` can be below 2. The docstring does
   not use this inequality, so the registered analysis is unaffected; the
   paper's small-step reading (`≤ 2α`) relies on it and is now proved.
5. **Tolerance and precision.** The registered analysis scores in float32
   and adds `1e-6` to the right-hand side; Proposition 2 is exact in real
   arithmetic, and the tolerance covers roundoff at numerical ties.

## 11. Verification

Command (run on 2026-09-25, `.venv` with NumPy 2.5.2, pytest 9.0.2):

```text
.venv/Scripts/python -m pytest tests/test_decision_change_condition.py tests/test_negative_paper_gate_n3_encoder_monte_carlo.py -q
```

Result: `27 passed` in about 7 s. No model, dataset, saved result, or
registered noise stream is read by either file; all geometry is synthetic and
seeded (`SEED = 20260925`).

### `tests/test_decision_change_condition.py` (17 tests)

| Test | What it checks | Outcome |
|---|---|---|
| `test_per_draw_identity_is_exact[K, norm]` (8 cases: `K ∈ {10, 100}`, `d = 512`, `‖w‖ ∈ {0.0025, 0.04, 0.5, 2.0}`) | Proposition 1 on 2,000 random unit `z` per case: `score'_k = (s_k + c)/n_k`, the pairwise margin formula (1) for all ordered pairs, and identity (2) | maximum deviations at or below `1e-12` (scores, margins) and `1e-11` (normalizer identity) |
| `test_normalizer_difference_lemma_including_adversarial_translations` | `|n_j − n_k| ≤ min{1, ‖w‖} ‖t_j − t_k‖` and identity (2) on more than 10,000 (pair, `w`) combinations, including `w = −β t_k`, `w = −β t_j`, `w = −t_k + λ m̂`, and `w = −β m̂` | zero violations; more than 100 of the combinations have `n_j + n_k < 2`, where the Cauchy-Schwarz route would not apply |
| `test_normalizer_difference_lemma_is_tight` | the equality case of Section 4 (`‖t_j − t_k‖ = 6/5`, `λ = 7/20`) | equality to `1e-9`; `n_j + n_k < 2` there |
| `test_necessary_condition_holds_on_every_observed_flip` | 12,000 `(z, w)` trials (three generation modes: random `z` and `w`; near-tie `z` with random `w`; near-tie `z` with `w` mostly along `t_j − t_k`); on every observed top-1 flip: the full chain (3) with both companion forms, the registered form, the top-1 form, `‖t_j − t_k‖ ≤ 2`, and the direction statement | 751 flips observed, present in all eight `(K, ‖w‖)` cells (`K = 10`: 84, 72, 87, 100; `K = 100`: 93, 83, 120, 112 at norms 0.0025, 0.04, 0.5, 2.0); zero violations; the largest observed ratio `(s_j − s_k)/|n_j − n_k|` among flips was about 0.93 |
| `test_per_class_scale_extension_on_tangent_translations` | Section 7 on tangent-form banks with `a_k > 0`: the registered per-class-scale inequality and the sharper intermediate bound on every observed flip | more than 50 flips, zero violations |
| `test_orthogonal_projection_translation_never_changes_the_argmax[K]` (2 cases) | Corollary 1 with `p` built as in `banks.py:218-223` (thresholded SVD) and cross-checked against a QR basis of the `K − 1` differences `t_k − t_K` (agreement `1e-9`); five scales of `p` (coefficients 0.25, 0.5, 1 and rescalings to `‖w‖ = 1` and `3`); 10,000 `z` per case: 5,000 random and 5,000 near-tie with margins from `1e-8` to `1e-2` | `max_{i,j} |⟨t_i − t_j, w⟩| ≤ 1e-10`, normalizer spread `≤ 1e-10`, zero argmax changes, identical vote counts |
| `test_orthogonal_projection_float32_flips_are_confined_to_numerical_ties` | float32 scoring of 4,000 near-tie draws (margins `1e-9` to `1e-3`) under an orthogonal translation | every float32 flip has exact original margin `≤ 1e-5` |
| `test_flip_budget_bounds_brute_force_counts` | Corollary 2 with `E_F` computed exactly as `d1_08_n3c_predictions.py:197-201`; 2 class counts × 4 norms × 3 item types (`already`, `split`, `mixed`) × 512 draws, against brute-force translated counts, with the registered 512-draw cut-off used as a fixed `k_min` | `#flips ≤ E_F ≤ coarse budget`; every flipped draw eligible at its actual pair; `max_i c'_i ≤ k_0 + E_F`; `|c'_i − c_i| ≤ E_F`; all three partition classes occurred and brute force agreed with every "unattainable" prediction and with the `k_0 + E_F < cutoff` implication for every cutoff on a grid of 16 |
| `test_two_sided_centering_can_flip_beyond_every_text_only_bound` | the explicit instance of Section 8 and the two-sided identity | text-only translation keeps class 1; the two-sided operator selects class 2 although the original margin exceeds `‖w‖ ‖t_1 − t_2‖ > |n_1 − n_2| = 0`; identity exact to `1e-12` |

### `tests/test_negative_paper_gate_n3_encoder_monte_carlo.py` (10 tests)

Frozen cases from `configs/negative_paper_gate_n3_theory_v1.json`
(`witness_kappa_gt_1`, `witness_kappa_lt_1`, `witness_small_q`), 200,000
draws `u ~ N(0, σ²)` per case with seed `20260925 + case index`,
`f(u) = normalize(z0 + u(e1 + κ e2))`, `Φ` through `math.erf`.

| Test | What it checks | Outcome |
|---|---|---|
| `test_frozen_config_is_non_authorizing_and_unchanged_in_shape` | the config still has three cases, the registered proposition id, the encoder string, and every authority flag `False` | pass |
| `test_empirical_class_probabilities_match_phi[case]` (3) | empirical positive-class frequencies against `Φ(q/σ)` and `Φ(q/(κσ))` within 4 standard errors; hard labels equal `sign(q + u)` and `sign(q + κu)` on every draw with argument above `1e-9`; the two banks differ by far more than Monte Carlo error on the same draws, in the direction given by `κ` | pass; the largest absolute deviation over the six checks was about 2.0 standard errors |
| `test_plug_in_population_radii_match_closed_form[case]` (3) | `σ Φ⁻¹(p̂)` against `q` and `q/κ` within a 4-standard-error delta-method band (plug-in population quantity, not a certificate) | pass |
| `test_two_point_reference_set_gives_identical_gap_vectors_and_correct_labels[case]` (3) | reference set `u = 0` (positive) and `u = −L` (negative) with `L = 2 max(q, q/κ)`: one class-balanced image mean for both banks, bit-identical prototype means and gap vectors, both inputs classified correctly by both banks, identical clean margin and prototype correlation at `u = 0` | pass |

### LaTeX

Both blocks compile under a scratch IEEEtran wrapper with the same preamble as
`paper/satml2027/main.tex` (`\newtheorem` environments, `\Normalize`,
`\argmax`, amsthm `proof`); the labels `prop:identity`, `prop:flip`,
`cor:chowers`, `cor:budget`, `rem:twosided`, `app:proofs` resolve, and the
displays were split to fit the two-column width. The full manuscript build
additionally needs `generated/macros.tex` from the asset generator.

## 12. What this note does not claim

- It does not claim that any operator is ineffective, effective, or
  equivalent to no correction on a population; Corollary 1 is the only exact
  invariance, and it concerns the projected-gap family only.
- It does not claim a bound on the two-sided operator, on class-conditioned
  rows (`noisy_class_mean`, `lowrank_tangent_r8`), on prompt or image-side
  methods, or on FSS/Smoothed-Embedding certificates.
- It does not turn `E_F`, `k_0 + E_F`, or any margin into a certificate; the
  registered stop rules (no certificate issued, no candidate selected,
  EXP-017 unaltered) are unchanged.
- It does not use, inspect, or predict any outcome of the running EXP-019A/B
  or N3C queue.

## 13. Files

- This note: `research/DECISION_CHANGE_CONDITION.md`
- Paper: `paper/satml2027/sections/theory_decision_change.tex`,
  `paper/satml2027/sections/appendix_proofs.tex`
- Tests: `tests/test_decision_change_condition.py`,
  `tests/test_negative_paper_gate_n3_encoder_monte_carlo.py`
- Registered statements documented here: `configs/satml2027/n3c_v3.json`
  (P1, P2, P7), `satml2027/day1/d1_08_n3c_predictions.py`,
  `satml2027/common/banks.py` (`operator_parameters`, `normalizer_statistics`)
- Saved outcome consistent with Corollary 1:
  `artifacts/negative_paper/exp017_scientific_soundness_audit_v1.json`
