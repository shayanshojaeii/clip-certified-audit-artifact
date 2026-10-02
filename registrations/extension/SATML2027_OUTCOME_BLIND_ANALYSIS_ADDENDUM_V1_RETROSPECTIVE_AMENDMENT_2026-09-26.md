# Retrospective amendment to the outcome-blind analysis addendum V1 (2026-09-26)

Status: AMENDMENT OF THE APPROVAL PROCEDURE ONLY; THE ADDENDUM V1 TEXT, ITS HASH AND THE REGISTERED PRIMARY ANALYSIS ARE UNCHANGED.
Trigger: correction G1 of the reviewer-bundle V3 audit (`research/independent_review/reviewer_bundle_V3_critical_audit_2026-09-26.pdf`, section 9), recorded in D-135.

## 1. Why this amendment exists

The addendum `research/SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.md` was written and hashed before the EXP-019 outcomes were opened, and it asked for an approval asserting `outcome_blind_at_approval: true`. The EXP-019 outcomes were opened on 2026-09-25 at the owner's instruction, before any approval (D-127). No review can now be outcome-blind, so the approval form in the addendum's section 0.1 cannot be signed truthfully. This amendment replaces that approval form with a retrospective review and leaves every other part of the addendum as frozen.

## 2. Three separate times

| Event | When | Record |
|---|---|---|
| Addendum text and code frozen, before any EXP-019 outcome existed | 2026-09-25 | D-118; addendum SHA-256 `0d5664c86011ecf464a702611275deb46d249598c623ca4d3d20ac8ccb16ccd1`; `d1_09_sensitivity_analysis.py` `a925153f6c99185b13f60e3c5285218fbcccece19cb42bdb0d32b0bc5949dc67`; `d1_10_tf32_preparation_adjudication.py` `00f74de18dc762100a2dc0516b554b5c30c4724e0313a4e818ea63d8af6d3322`; primary analysis `d1_07_analyze.py` `546e34a5eb986b341a636581ea9726828176e09df8b864b60ccc23df6bbc63d2` |
| Outcomes accessible and inspected | 2026-09-25 | D-127, owner-instructed; the registered primary analysis ran unchanged (D-127, D-128) |
| Independent methodological review | Not yet performed; any review from now on is retrospective | This amendment |

A plan drafted before unblinding keeps that historical status. A later review can judge its methods sound. Neither makes the later review outcome-blind, and nothing here backdates a signature or alters the historical freeze.

## 3. What does not change

- The addendum file keeps its bytes and its SHA-256 above.
- The registered primary analysis `satml2027/day1/d1_07_analyze.py` stays unchanged, and its outputs stay the primary results.
- Items A1-A9 of the addendum keep their content: bands, strata, pooled descriptions, multiplicity statement, the TF32 adjudication rule, reporting language and power statement.
- No new sampling, no equivalence, causal or confirmatory claim, and no final-test access are authorized.

## 4. What changes: the approval schema

The outcome-blind approval file `research/SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.approved.json` is retired. If it exists, `d1_09_sensitivity_analysis.py` refuses to run. In its place, a retrospective review is recorded in `research/SATML2027_OUTCOME_BLIND_ANALYSIS_ADDENDUM_V1.retrospective_review.json` with exactly these fields:

| Field | Required value |
|---|---|
| `status` | `INDEPENDENTLY_REVIEWED_RETROSPECTIVELY_ADDENDUM_V1` |
| `addendum_sha256` | SHA-256 of the unchanged addendum V1 (section 2) |
| `retrospective_amendment_sha256` | SHA-256 of this file as reviewed |
| `sensitivity_source_sha256` | SHA-256 of `satml2027/day1/d1_09_sensitivity_analysis.py` as reviewed (its guard implements this schema) |
| `tf32_adjudication_source_sha256`, `primary_analysis_source_sha256` | SHA-256 of `d1_10` and of `d1_07` as reviewed (optional; checked if present) |
| `new_sampling_authorized`, `equivalence_claim_authorized`, `confirmatory_claim_authorized`, `causal_claim_authorized`, `final_test_authorized` | all `false` |
| `outcome_blind_at_approval` | `false` |
| `outcomes_accessed_before_review` | `true` |
| `unblinding_decision` | `D-127` |
| `unblinding_date` | `2026-09-25` |
| `review_date` | ISO date, not before `2026-09-25` |
| `reviewer`, `reviewer_conclusion` | free text; the reviewer states whether the review was AI-assisted |

The guard in `d1_09` checks every field above, refuses a retired outcome-blind approval, and refuses to overwrite an existing sensitivity output. Its outputs carry the label "secondary sensitivity, reviewed retrospectively after unblinding (D-127)".

## 5. What a retrospective review can and cannot establish

It can establish that items A1-A9 are unambiguous and methodologically sound, and that the code implements them as stated. It cannot establish that the analysis choices were made without knowledge of the outcomes after 2026-09-25, nor that the reviewer was blind to them. The sensitivity analyses therefore remain secondary: they are reported beside the unchanged registered primary analysis and never replace it. The manuscript does not report any `d1_09` output; if one is added later, it must carry the label above.
