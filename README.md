# Artifact: Auditing Shared Modality-Gap Corrections for Certified Correctness in Frozen CLIP

Anonymous artifact for an IEEE SaTML 2027 submission (double-blind review).

This repository holds the artifact described in the paper's Open Science section. It is assembled from
hash-identified files and is completed by the artifact deadline (2 October 2026, AoE); after that date it is not
edited during review. A manifest (`MANIFEST.sha256.json`) lists every file with its SHA-256.

## Layout (as described in the paper)

- `registrations/`: protocol, pre-analysis and extension registrations with hashes
- `results/exp017/`: per-cell sufficient statistics, candidate metadata and bank hashes of the audit
- `analysis/`: metric-semantics audit, simultaneous-band output and configuration, exact-count adjudication
- `paper/`: manuscript source, generators, generated tables and figure data, tests, claim lint
- `theory/`: decision-change note and verification tests
- `control/`: positive-control release
- `extension/`: code, registrations, per-item vote counts, prototype banks, registered analysis outputs, reconstruction package
- `followup/`: registrations, stage approvals (approver names redacted), per-item vote counts with custody manifests, registered analysis outputs
- `review/`: review ledger

Reproduction needs no GPU: see `REPRODUCE.md` (added with the files).
