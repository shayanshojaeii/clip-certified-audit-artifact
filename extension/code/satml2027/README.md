# SaTML 2027 submission bundle - Day 0 and Day 1

Everything needed to close Phase-4 custody, register the extension, and run it on
**server A (1x RTX 4090)** and **server B (4x RTX 3090)**.

Install into the project root so that `results/`, `configs/`, `data/` and the
existing `interventions/` package are siblings of `satml2027/`.

```
satml2027/
  common/    hashing, seeds, certification, prompts, models, banks, device planning
  day0/      custody, item draw, registrations, artifact skeleton
  day1/      preflight, development, controls, banks, plan, worker, merge, analysis
  launch/    prepare + launch helpers
```

Only `numpy` and `scipy` are needed for the analysis; `torch`, `torchvision` and
`open_clip` are needed for anything that touches a GPU.

---

## Where each step runs

The project root (git repo, EXP-016/EXP-017 results, Phase-3 splits) lives on the
**laptop**; the GPUs live on **server A** (1x RTX 4090) and **server B** (4x RTX 3090).

| Steps | Host | Needs |
|---|---|---|
| Day 0 (`d0_01` ... `d0_04`) | **laptop** | numpy, scipy, the repo, the Phase-4 bundle, the CIFAR-10 python archive |
| Day 1 preparation and sampling (`d1_01` ... `d1_05`) | **servers A and B** | torch, torchvision, open_clip, the datasets, the checkpoints |
| Merge and analysis (`d1_06` ... `d1_08`) | **laptop** | numpy, scipy only |

Day 0 needs no GPU and no torch: the CIFAR-10 item draw reads the pickled
`data_batch_*` files directly when torchvision is absent, in exactly torchvision's
order.  Run `d0_01` wherever the 809 MB Phase-4 bundle actually sits; if that is a
server, run it there and copy `results/satml2027/custody/` back.

**What moves between hosts**

```
laptop -> servers   satml2027/  configs/satml2027/  results/satml2027/items/     (a few MB)
servers -> laptop   results/satml2027/EXP-20260920-019A|B/  and  .../N3C/        (a few hundred MB)
```

Both servers additionally need the datasets and the five open_clip checkpoints;
those are fetched on the servers and never travel over your connection.

## Day 0 (tonight, CPU only, ~15 minutes, on the laptop)

```bash
# 1. close Phase-4 custody and prove the hash chain
python satml2027/day0/d0_01_verify_custody.py \
  --phase3-assignments artifacts/day14/phase3_splits_v1/assignments.csv \
  --phase4-results     results/phase4/EXP-20260917-020-FARLA-FULL \
  --phase4-config      configs/phase4/farla_full_v1.json \
  --ledger             artifacts/ledger/item_roles.jsonl \
  --out                results/satml2027/custody --copy-config
# expect: status PASS, reserved_remaining {cifar100: 41480, eurosat: 24350}

# 2. draw and register the evaluation / development / control-train items
python satml2027/day0/d0_02_draw_items.py \
  --phase3-assignments artifacts/day14/phase3_splits_v1/assignments.csv \
  --phase4-results     results/phase4/EXP-20260917-020-FARLA-FULL \
  --data-root data --out results/satml2027/items
# CIFAR-10 must already be downloaded into data/ (train split only).

# 3. write and hash the three preregistrations
python satml2027/day0/d0_03_make_registrations.py \
  --items results/satml2027/items --out configs/satml2027 --scope full
git add configs/satml2027 results/satml2027/items results/satml2027/custody && \
  git commit -m "preregister N3C-20260920-V3 and EXP-20260920-019A/B"   # timestamp the hashes

# 4. artifact skeleton for the SaTML Open Science requirement
python satml2027/day0/d0_04_init_artifact_repo.py --out artifact --bundle satml2027
```

**The registrations must be committed before any sampling starts.**  The worker
recomputes the canonical hash of the registration it is given and refuses to run
if the file has been edited.

---

## Day 1

### 1. Preflight on *both* servers (~10 min each)

```bash
python satml2027/day1/d1_01_preflight.py --server A --out results/satml2027/preflight   # server A
python satml2027/day1/d1_01_preflight.py --server B --out results/satml2027/preflight   # server B
```

Then compare `rng_digests` between the reports. The amended planner keeps every
scientific cell on one server regardless; shards may span GPUs only within that
server. A digest mismatch is therefore a reported environment difference, not
permission to split one cell across environments.

The report also measures encodes/s per model and reports the precision
equivalence check (float32 versus TF32 / autocast) that the paper cites.

### 2. Development artifacts, control banks, candidate banks

```bash
# custody proof: the ported bank rule reproduces the saved EXP-017 banks exactly
python satml2027/day1/d1_03_build_banks.py --verify-against-exp017 \
  --exp016 results/EXP-20260906-016 --exp017 results/EXP-20260906-017

./satml2027/launch/day1_prepare.sh "$PWD" <FROZEN_EXP016_TEMPERATURE>
```

`day1_prepare.sh` runs, for every cell: `d1_02_build_development.py` (prototypes,
clean and noisy development features, the two proposal directions),
`d1_03b_train_controls.py` (the learned shared translation and the rank-8 tangent
adapter, plus the control-train feature cache) and `d1_03_build_banks.py` (the 18
frozen candidates plus four controls).

Pass the **frozen EXP-016 lower-tail temperature** (`0.05`). The amended worker
allows only `--direction-builder project` and calls the exact Phase-3 interfaces;
there is no fallback implementation.

Then copy the small artifacts to the other server:

```bash
rsync -a results/satml2027/{items,banks,development,controls} serverB:/path/to/project/results/satml2027/
rsync -a configs/satml2027 serverB:/path/to/project/configs/
```

### 3. Plan and launch

```bash
python satml2027/day1/d1_04_plan.py \
  --registrations configs/satml2027/exp-20260920-019a.json configs/satml2027/exp-20260920-019b.json configs/satml2027/n3c_v3.json \
  --preflight results/satml2027/preflight --servers A:1 B:4 --out results/satml2027/plan

./satml2027/launch/day1_launch_all.sh A 1     # on server A
./satml2027/launch/day1_launch_all.sh B 4     # on server B
```

The planner assigns each complete cell to one server, then splits that cell only
over GPUs on the selected server. With
the fallback throughput model the full scope is ~111 GPU-hours and ~22 hours of
wall clock across the five cards; the plan is recomputed from the *measured*
preflight numbers, so read `plan.json` before committing to the schedule.

Every shard checkpoints every 10 items and resumes where it stopped, so a crash,
a reboot or a pre-emption costs at most ten items.

**Scope ladder** (descend only, and record the reason and time before running
the affected cells): `full` ~111 GPU-h -> `no_sigma_012` ~82 GPU-h ->
`core` ~35 GPU-h.  Re-run `d0_03_make_registrations.py --scope ...` to descend.

### 4. N3c, the fresh-noise diagnostic

```bash
# N3C is included in the amended planner above. Its launch lines automatically
# add --item-role development --store-features for exactly four registered cells.
python satml2027/day1/d1_08_n3c_predictions.py --features results/satml2027/N3C-20260920-V3/cells \
  --banks results/satml2027/banks --registration configs/satml2027/n3c_v3.json --out analysis/n3c
```

### 5. Merge and analyse

```bash
rsync -a serverB:/path/to/project/results/satml2027/EXP-20260920-019A/ serverB_pull/EXP-20260920-019A/
python satml2027/day1/d1_06_merge_verify.py --registration configs/satml2027/exp-20260920-019a.json \
  --roots results/satml2027/EXP-20260920-019A serverB_pull/EXP-20260920-019A \
  --out results/satml2027/merged/EXP-20260920-019A
python satml2027/day1/d1_07_analyze.py --merged results/satml2027/merged/EXP-20260920-019A \
  --registration configs/satml2027/exp-20260920-019a.json --out analysis/019a
```

`d1_06` fails loudly if the shards do not tile the item list exactly once, if any
item is unfinished, or if any row of votes does not sum to the registered budget.
`d1_07` is torch-free and is the script shipped in the artifact.

---

## Design decisions that are custody-relevant

| Decision | Why |
|---|---|
| noise generated **on the GPU** in fixed blocks of 64 draws, seeded per (item, block) | nothing but vote counts crosses PCIe, and the forward batch size can be tuned per GPU without changing a single realized draw, so shards are bit-identical to a single-process run |
| seed payload includes model, dataset, **sigma**, item and stream | the sigma sweep uses independent noise streams rather than one stream rescaled |
| **float32 with TF32 disabled** | the winner-runner-up margin is of order 1e-3 per draw, so reduced-precision matmuls are not used for the confirmation stream; `--allow-tf32` exists only for benchmarking, and the preflight quantifies what it would cost |
| every bank scored on the **same** encoded features | every paired contrast uses common random numbers at the draw level, which is what makes a +0.15pp interval meaningful |
| exact `(a_k, n_k)` recorded when each bank is built | the N3c containment test and the P7 flip budget are exact rather than reverse-engineered from the banks (on a synthetic check the recorded form tightens the median flip budget from 254 to 190 draws) |
| `d1_03_build_banks.py --verify-against-exp017` | proves the ported construction rule still reproduces the audited banks |
