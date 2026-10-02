# Review ledger

Each row is one review of the work: its date, the exact version reviewed, its scope and verdict, the reviewer's role,
and whether the review states that it was AI-assisted. Reviewers are named by role in this anonymous version; "first reviewer" and so on number the reviewers within one review round.

**Who reviewed.** Every review listed here was run by a member of the author team. Several team members ran their
reviews with AI assistants; the "AI assistance" column records this as each review states it ("not stated" means the
review does not say). The project owner confirmed in writing, for the team, that team members re-evaluated or replayed
every result reported in the paper (column "Confirmed").

**Not listed.** Automated checks of the experiment-control code by AI coding agents, which the paper does not report,
are not reviews by team members and are not listed.

| # | Date | Subject (immutable version) | Scope | Verdict | Reviewer | AI assistance | Confirmed |
|---:|---|---|---|---|---|---|---|
| 1 | 2026-09-03 | Gate-2 fallback saved results | Saved-result chain | Verified | Second author | Not stated | Owner, for the team |
| 2 | 2026-09-03 | Gate-2 production code, round 2 | Certification and attack code | Passed with hardening | Second author | Not stated | Owner, for the team |
| 3 | 2026-09-03 | Gate-2 hardening and full test suite | Unconditional sign-off | Pass | Second author | Not stated | Owner, for the team |
| 4 | 2026-09-10 | Audit protocol, revision 8.1 (`9800552B...`) | Pre-sampling review | Pass | Second author | Not stated | Owner, for the team |
| 5 | 2026-09-11 | Audit development stage and authorization of the validation stage | Independent reconstruction of the development stage | Pass | Second author | Not stated | Owner, for the team |
| 6 | 2026-09-11 | Audit execution readiness | Preprocessing mapping, prototype reuse, smoke runs | Pass | Second author | Not stated | Owner, for the team |
| 7 | 2026-09-11 | Audit launcher amendment | Dynamic-import correction | Pass | Second author | Not stated | Owner, for the team |
| 8 | 2026-09-11 | Audit model-loader amendment | Loader correction | Pass | Second author | Not stated | Owner, for the team |
| 9 | 2026-09-11 | Audit runtime-asset amendment V2 | Clean-model mapping and recovery | Pass | Second author | Not stated | Owner, for the team |
| 10 | 2026-09-12 | Audit result chain | Reconstruction of the saved result: banks, counts, certificates, selection | Scientific negative verified | Team member | Not stated | Owner, for the team |
| 11 | 2026-09-13 | Scope gate of a discontinued track | Scope review | Approved with corrections; sampling prohibited | Team member | Not stated | Owner, for the team |
| 12 | 2026-09-15 | Metric-semantics bundle V1 | Metric semantics and saved-data reconstruction of the audit | Pass | Second author | Not stated | Owner, for the team |
| 13 | 2026-09-23 | Extension pre-sampling amendment V4 (commit `7c26704e...`) | Extension queue, worker, merge, fresh-noise validator | Pass | Team member | Not stated | Owner, for the team |
| 14 | 2026-09-24 | Simultaneous-analysis bundle V2 (`003c4cf9...`) | One read-only execution of the complete-family band | Pass | Team member | Not stated | Owner, for the team |
| 15 | 2026-09-24 | Non-identification bundle V2 (`5e045fa4...`) | Non-identification theory | Pass; wording pass with correction | Team member | Not stated | Owner, for the team |
| 16 | 2026-09-26 | Follow-up V1 package and reviewer bundle V1 | Pre-sampling review; manuscript | Follow-up V1 fail | Team member | Not stated | Owner, for the team |
| 17 | 2026-09-26 | Follow-up V1 package and reviewer bundle V1 | Critical audit | Withhold the follow-up; manuscript pass with corrections | Team member | Cited by row 19 as an AI-assisted report | Owner, for the team |
| 19 | 2026-09-26 | Reviewer bundle V3 (`772a436a...`) | Critical audit: manuscript, evidence, follow-up | Corrections on two items; passes elsewhere, including a byte-identical replay of the band; follow-up stage 1 hold | Team member | AI-assisted (self-described) | Owner, for the team |
| 20 | 2026-09-26 | Reviewer bundle V3 (`772a436a...`) | Text review: custody, replays, follow-up | Custody pass; band replay pass (byte-identical); theory pass; manuscript mechanics pass | Team member | Not stated | Owner, for the team |
| 21 | 2026-09-27 | Reviewer bundle V4 (`32f3cfc7...`) | Full audit including raw replays: extension custody (76/76 files) and reconstruction (704 cell rows exact; contrasts within 5.6e-17), fresh-noise diagnostic (4 cells exact), positive-control custody (61/61 raw files) and reconstruction (aggregates within 1.11e-16; five planned intervals exact), band, theory, manuscript, follow-up V4 (194 tests) | Pass with corrections | Team member (first reviewer) | AI-assisted (self-described) | Owner, for the team |
| 22 | 2026-09-27 | Reviewer bundle V4 (`32f3cfc7...`) | Full audit from the bundle alone | Pass with corrections; raw replays of the extension and the positive control not possible from the bundle | Team member (second reviewer) | Not stated in the review | Owner, for the team |
| 23 | 2026-09-27 | Reviewer bundle V5 (`924926bf...`) | Full audit: packages, hashes, tests, source and PDF, bibliography, claim wording, anonymity tooling, venue rules | Pass with corrections | Team member (first reviewer) | AI-assisted (self-described) | Owner, for the team |
| 24 | 2026-09-27 | Reviewer bundle V5 (`924926bf...`) | Full audit from the bundle | Scientific chain pass; manuscript pass with corrections; follow-up package exact (194 tests) | Team member (second reviewer) | Not stated | Owner, for the team |
| 25 | 2026-09-27 | Reviewer bundle V5 (`924926bf...`) | Technical and scientific integrity audit with replays and adversarial probes | Paper hold until seven bounded corrections (applied); band, theory, generated assets and custody chains pass; follow-up stage 1 hold on two validator defects (repaired) | Team member (third reviewer) | Not stated | Owner, for the team |
| 27 | 2026-09-27 | Follow-up V4.1 delta package (`b7f6be8e...`) | Adversarial review of the code delta, probes, template, commit binding, host tooling and registration regeneration | Both repairs pass; stage 1 hold on two operational items (corrected) | Team member | Not stated | Owner, for the team |
| 29 | 2026-09-28 | Follow-up result bundle V1 (`bb0fb1ad...`) and manuscript V7 (`661e4275...`) | Technical and scientific audit, including an exact-integer reconstruction of the follow-up's 385 primary-radius contrasts and 2,310 raw tail fields | Custody, Cohen arithmetic, saved results and mathematics sound; nine raw tail fields outside the primary family affected by floating-point boundary ties, no class changes; manuscript corrections | Team member (first reviewer of this round) | AI-assisted (as recorded) | Owner, for the team |
| 30 | 2026-09-28 | Follow-up result bundle V1 and manuscript V7 | Review of results and manuscript | Pass with corrections | Team member (second reviewer of this round) | AI-assisted (as recorded) | Owner, for the team |
| 31 | 2026-09-28 | Follow-up result bundle V1 and manuscript V7 | Scientific-interpretation review | Wording corrections | Team member (third reviewer of this round) | AI-assisted (as recorded) | Owner, for the team |
| 32 | 2026-09-29 | Manuscript bundle V10 (`e83af6ee...`) | Technical audit and first-round review: archive and manifests, tests (90 passed, 2 documented skips), claim lint, byte-identical macro regeneration, follow-up macro replay | Three must-fix items (applied) and framing advice | Team member (second reviewer) | Not stated | Owner, for the team |
| 33 | 2026-09-29 | Manuscript bundle V10 (`e83af6ee...`) | First-round review: archive, clean rebuild, tests (79/79), claim lint, submission gate | Borderline to weak accept; statistical-presentation advice (applied in part) | Team member (third reviewer) | Not stated | Owner, for the team |
| 34 | 2026-09-29 | Manuscript bundle V11 (`f7139c9a...`) | First-round review | Borderline with a path to acceptance; framing and wording advice (applied) | Team member (second reviewer) | Not stated | Owner, for the team |
| 35 | 2026-09-29 | Manuscript bundle V11 (`f7139c9a...`) | First-round review: archive, clean build, tests (30/30; 79/79), lint, gate | Revision; margin, pooled mixture, controls, compute disclosure (applied) | Team member (third reviewer) | Not stated | Owner, for the team |
| 36 | 2026-09-29 | Manuscript bundle V11 (`f7139c9a...`) | Technical, scientific and mathematical integrity audit: package, registrations, generated assets, certificate semantics, independent reconstruction of predictions X3-X5, exact bootstrap-boundary audit, mathematics | Two preregistration-wording corrections (X4 rule, X5 status) and a discordance disclosure (applied); mathematics pass | Team member (fifth reviewer) | AI-assisted (self-described) | Owner, for the team |
| 37 | 2026-09-29 | Manuscript bundle V11 | Framing review | One orientation sentence (applied) | Team member (first reviewer) | Not stated | Owner, for the team |
| 38 | 2026-09-29 | Manuscript bundle V11 | Presentation review | Figure, caption and documentation fixes (applied) | Team member (fourth reviewer) | Not stated | Owner, for the team |
| 39 | 2026-10-02 | This artifact | Exact-integer recomputation of the follow-up bootstrap tails (`followup/exact_boundary_recomputation/`) and replay of every follow-up output from the staged artifact | The nine fields of row 29 reproduced exactly; no class change; full replay pass | Project owner, with an AI coding assistant | AI-assisted (coding assistant) | Owner |

Rows 18, 26 and 28 of the internal ledger are the automated code checks described above and are omitted; the numbering
is kept so that the internal and released ledgers can be matched.
