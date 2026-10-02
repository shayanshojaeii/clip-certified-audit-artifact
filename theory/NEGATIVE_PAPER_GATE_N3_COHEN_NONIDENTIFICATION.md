# Gate N3: Exact Cohen Non-Identification Witness

**Status:** independently verified; paper-facing reference-distribution correction incorporated  
**Claim touched:** C1  
**Purpose:** show, without importing FSS-specific terms, that a fixed clean global
modality-gap vector does not determine a Cohen randomized-smoothing radius.

## 1. Scope

This is a synthetic binary construction. It proves non-identification: two
normalized prototype systems can have the same fixed encoder, clean image
feature, complete global gap vector, clean margin, and prototype correlation,
yet have different population Cohen radii. It does **not** establish prevalence
in CLIP, the size of an empirical effect, or failure of every gap correction.

The radius below is the population Cohen radius. A finite Monte Carlo
certificate instead uses an exact lower confidence bound on the selected class
probability and is random and conservative.

## 2. Construction

Let `e1,e2,e3` be the standard basis of `R^3`. Choose

```text
0 < gamma < 1,   s = sqrt(1-gamma^2),
b > 0,           d > 0,
kappa > 0,       kappa != 1,
sigma > 0.
```

Define the clean feature

```text
z0 = (b e1 + b e2 + d e3) / sqrt(2b^2+d^2)
q  = b / sqrt(2b^2+d^2) > 0.
```

Use the same scalar-input encoder for both prototype banks:

```text
f(u) = Normalize(z0 + u(e1+kappa e2)).
```

The two binary banks are

```text
Bank 1: t1+ = gamma e3 + s e1,   t1- = gamma e3 - s e1
Bank 2: t2+ = gamma e3 + s e2,   t2- = gamma e3 - s e2.
```

All four prototypes are unit vectors. Both banks have the same prototype mean
`m=gamma e3`. Let `P_X` be any common fixed image-reference distribution for
which the mean feature exists, and define

```text
mu_I = E_{X~P_X}[f(X)],
gap  = prototype mean - mu_I.
```

The encoder and `P_X` are shared, so `mu_I` is identical for both systems.
Because their prototype means are also identical, both systems have exactly the
same complete global gap vector `m-mu_I`. The original verifier's choice
`P_X=delta_0` is the valid degenerate special case `mu_I=z0`; the proposition
does not depend on that special choice.

For an explicit class-balanced finite reference set, take two inputs

```text
u_plus = 0,
u_minus = -L,  with L > max(q, q/kappa),
```

and assign positive and negative labels respectively. Both banks classify both
reference inputs correctly. With uniform class weights,

```text
mu_I = [f(0)+f(-L)]/2,
```

which is common to both systems; therefore the empirical class-balanced global
gap is identical as well.

## 3. Shared clean geometry

The positive-class clean margins are identical:

```text
z0^T(t1+ - t1-) = 2sq = z0^T(t2+ - t2-) > 0.
```

The within-bank prototype correlations are also identical:

```text
t1+^T t1- = gamma^2-s^2 = t2+^T t2-.
```

Thus the clean positive prediction at the certified query `u=0` is unique in
both banks, and the common-reference global gap, clean margin, and prototype
correlation are all held fixed.

## 4. Different Cohen probabilities and radii

Because normalization divides all scores by the same positive denominator, the
binary decision signs are

```text
Bank 1 predicts + iff q+u > 0;
Bank 2 predicts + iff q+kappa*u > 0.
```

With `u=epsilon ~ Normal(0,sigma^2)`, continuity gives zero tie probability and

```text
p1 = Phi(q/sigma),
p2 = Phi(q/(kappa sigma)).
```

For a binary smoothed classifier, `p_minus=1-p_plus`. Cohen et al.'s population
radius therefore simplifies to

```text
R = (sigma/2)[Phi^-1(p_plus)-Phi^-1(p_minus)]
  = sigma Phi^-1(p_plus).
```

Consequently,

```text
R1 = q,       R2 = q/kappa.
```

They differ whenever `kappa != 1`, even though every clean quantity listed
above is identical. The missing information is how the fixed encoder's
perturbation direction meets each prototype decision boundary.

## 5. Regularity of the encoder

Let `v=e1+kappa e2` and `g(u)=z0+uv`. The third coordinate of `g(u)` is the
strictly positive constant

```text
h = d/sqrt(2b^2+d^2).
```

Therefore `||g(u)|| >= h > 0` for every real `u`; normalization is globally
defined. Moreover,

```text
f'(u) = [I-f(u)f(u)^T]v / ||g(u)||,
||f'(u)|| <= sqrt(1+kappa^2)/h.
```

The witness is smooth and globally Lipschitz. It does not rely on the
piecewise-constant FSS witness used in the earlier C1 corollary.

## 6. Proposition

> **Proposition (clean global-gap non-identification for Cohen smoothing).**
> For every admissible parameter choice above, the two normalized binary
> prototype systems share one smooth globally Lipschitz encoder and, under any
> common fixed image-reference distribution with an existing mean feature,
> have the same complete global gap vector. At the certified query `u=0` they
> also have the same clean image feature, positive clean pairwise margin, and
> within-bank prototype correlation. Under the same scalar Gaussian input
> perturbation they nevertheless have population Cohen radii `q` and
> `q/kappa`. Hence the clean global gap—even augmented by those clean pairwise
> summaries—does not identify the Cohen randomized-smoothing radius.

The proof is exactly the calculation in Sections 2--5. The second team member
independently rederived the construction and approved the mathematical claim;
the common-reference wording above incorporates the sole requested
paper-facing correction.

## 7. Engine boundary

- The shared normalized prototype geometry is certificate-independent.
- The class probabilities and radius in Section 4 use Cohen hard-label
  randomized smoothing only.
- `tau*`, FCSB, and the FSS stability/geometry radius decomposition are not used
  here and must not be attached to this proposition.
- Pautov Smoothed Embeddings is a different protected classifier and conversion;
  it is not used here.
- The older exact FSS example remains a separate corollary tied to the archived
  FSS v3 preprint.

## 8. Edge cases and exclusions

- `b=0` removes the strict clean prediction and gives a vacuous boundary witness.
- `d=0` removes the simple global nonzero-denominator proof.
- `gamma` outside `(0,1)` degenerates the two-prototype construction.
- `kappa<=0` changes the stated probability derivation; `kappa=1` makes the two
  radii equal.
- `sigma<=0` is not randomized smoothing.
- Multiclass CLIP probabilities are joint all-competitor events. This binary
  witness does not provide a scalar formula for arbitrary CLIP.

## 9. Verification artifacts

- Frozen cases: `configs/negative_paper_gate_n3_theory_v1.json`
- Independent-checkable implementation: `analysis/negative_paper_gate_n3.py`
- Regression tests: `tests/test_negative_paper_gate_n3_cohen_construction.py`
- Candidate paper block:
  `paper/generated/negative_paper_gate_n3_theory_candidate_v1.tex`
- Primary report:
  `artifacts/negative_paper/gate_n3_primary_verification_v1.json`

No scientific data are opened and no random samples are drawn by these checks.
