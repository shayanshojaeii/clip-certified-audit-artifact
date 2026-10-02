# Anonymous artifact (double-blind review)

This repository is the artifact of an anonymous submission. It holds the registrations, the released data, the
analysis outputs, the code and the manuscript source that the paper's Open Science section describes. Every file is
listed with its SHA-256 in `MANIFEST.sha256.json`. The repository is not edited during review.

## Layout

| Folder | Contents | Paper |
|---|---|---|
| `registrations/` | Audit protocol, decision rule, operator interfaces, prompts and the post-hoc analysis plan; extension and fresh-noise diagnostic registrations with hash sidecars and the outcome-blind analysis addendum; follow-up registrations with hash sidecars | Sections IV, VII, VIII; Table X |
| `results/exp017/` | The audit: per-cell Cohen sufficient statistics, candidate metadata, prototype banks, cell metrics, selection and summary; `development/` holds the development stage whose tensors built the banks | Sections IV-V |
| `analysis/` | Metric-semantics audit, complete-family simultaneous band (output and reconstruction script), exact-count gate adjudication and diagnostics, the independent reconstruction record | Section V, Appendix C |
| `theory/` | Decision-change note, non-identification note, theory configuration, numerical verification and tests | Section III, Appendices A-B |
| `control/` | Positive control: configuration snapshot, split indices, trained adapters, confirmation outputs, analysis report, and its custody package (manifest, reconstruction script and report) | Section VI |
| `extension/` | Extensions A and B: code, per-item vote counts of all 36 cells, prototype banks, custody report, receipts, recovery record, registered analysis outputs (including the fresh-noise diagnostic outputs), transfer manifests and the reconstruction package | Section VII, Appendix C |
| `followup/` | Follow-ups A-C: code, stage approvals (approver name replaced by a role; see `followup/approvals/REDACTION.json`), data preflight and bank manifest, per-item vote counts with run custody manifests, item lists, planning inputs, registered analysis outputs, and the exact-integer recomputation of the bootstrap tail probabilities (`exact_boundary_recomputation/`: 2,310 raw tail fields compared, nine differ, no direction or magnitude class changes) | Section VIII, Appendices D-E |
| `paper/` | Manuscript source as submitted, with the generated tables and figure data and the PDF; generators, replay, claim lint, submission gate, anonymity scanner and tests; `INPUT_MAP.json` and `stage_inputs.py` | whole paper |
| `review/` | Review ledger | Appendix E |

## Study names

| Study in the paper | Files |
|---|---|
| Audit (development and validation stages) | `results/exp017/`, `registrations/audit/`, `analysis/` |
| Positive control (pilot) | `control/` |
| Extensions A and B | `extension/vote_counts/EXP-20260920-019A/`, `.../EXP-20260920-019B/`, `registrations/extension/` |
| Fresh-noise diagnostic | `registrations/extension/n3c_v3.json`, `extension/registered_analysis/` |
| Follow-ups A, B and C | `followup/vote_counts/EXP-20260921-021A/`, `-021B/`, `-021C/`, `registrations/followup/` |

## Not included

The development and control-training caches of the extension (about 5.8 GB), the per-draw features of the fresh-noise
diagnostic (about 3.0 GB), the feature cache of the positive control (about 0.8 GB) and the prepared prototype banks
and preparation caches of the follow-up exceed the repository's limits. They are deposited in an archival repository
with the final version. Model weights and datasets are public third-party assets and are referenced by hash.

## Reproduction

See `REPRODUCE.md`. No GPU is needed.
