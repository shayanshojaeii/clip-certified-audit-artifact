#!/usr/bin/env bash
# Day 1 preparation: development artifacts, control banks, candidate banks.
# Run on whichever server has the datasets and checkpoints; the outputs are small
# and are rsynced to the second server before the sampling launch.
#
#   ./launch/day1_prepare.sh <PROJECT_ROOT> <NOISY_TEMPERATURE>
#
# The temperature is the frozen EXP-016 lower-tail temperature; read it from the
# Phase-3 protocol config, do not guess it.
set -euo pipefail
ROOT="${1:?usage: day1_prepare.sh PROJECT_ROOT NOISY_TEMPERATURE}"
TAU="${2:?missing frozen EXP-016 noisy-margin temperature}"
cd "$ROOT"

RESULTS=results/satml2027
ITEMS=$RESULTS/items
DATA=data
GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

MODELS_A=(openai-clip-vit-b32-quickgelu openai-clip-vit-l14-quickgelu)
MODELS_B=(openai-clip-vit-b16-quickgelu openclip-vit-b32-laion2b openai-clip-rn50-quickgelu)
DATASETS=(cifar100 eurosat cifar10)

run_cell () {  # model dataset sigma gpu registration
  local model="$1" dataset="$2" sigma="$3" gpu="$4" registration="$5"
  local prompt_config=""
  case "$dataset" in
    cifar100) prompt_config="configs/prompts/cifar100_openai_readme.json" ;;
    eurosat) prompt_config="configs/prompts/eurosat_openai_ensemble_v1.json" ;;
    cifar10) prompt_config="" ;;
    *) echo "unregistered dataset prompt mapping: $dataset" >&2; exit 2 ;;
  esac
  local prompt_args=()
  if [ -n "$prompt_config" ]; then prompt_args=(--prompt-config "$prompt_config"); fi
  CUDA_VISIBLE_DEVICES="$gpu" python satml2027/day1/d1_02_build_development.py \
      --model-id "$model" --dataset-id "$dataset" --sigma "$sigma" \
      --items "$ITEMS" --data-root "$DATA" --out "$RESULTS/development" \
      --registration "$registration" \
      --project-root "$ROOT" --noisy-temperature "$TAU" \
      "${prompt_args[@]}" \
      --direction-builder project
  CUDA_VISIBLE_DEVICES="$gpu" python satml2027/day1/d1_03b_train_controls.py \
      --registration "$registration" \
      --model-id "$model" --dataset-id "$dataset" --sigma "$sigma" \
      --items "$ITEMS" --data-root "$DATA" --development "$RESULTS/development" \
      --out "$RESULTS/controls" \
      --train-draws 16 --epochs 80 --batch-size 1024 --learning-rate 0.005 \
      --weight-decay 0.0001 --temperature 0.05 --clean-weight 0.5 --rank 8 \
      --noise-seed 20260920005 --optimizer-seed 20260920006 \
      --minibatch-seed 20260920007
  local cell="${model}__${dataset}__sigma${sigma}"
  python satml2027/day1/d1_03_build_banks.py --development "$RESULTS/development" --items "$ITEMS" \
      --cell "$cell" --registration "$registration" --out "$RESULTS/banks" \
      --controls "$RESULTS/controls/${cell}__controls.pt" \
      --control-train-features "$RESULTS/controls/${cell}__control_train_features.pt"
}

JOBS=()
for model in "${MODELS_A[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for sigma in 0.12 0.5; do JOBS+=("$model|$dataset|$sigma|configs/satml2027/exp-20260920-019a.json"); done
  done
  JOBS+=("$model|cifar10|0.25|configs/satml2027/exp-20260920-019a.json")
done
for model in "${MODELS_B[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for sigma in 0.25 0.5; do JOBS+=("$model|$dataset|$sigma|configs/satml2027/exp-20260920-019b.json"); done
  done
done
for model in "${MODELS_A[@]}"; do
  for dataset in cifar100 eurosat; do
    JOBS+=("$model|$dataset|0.25|configs/satml2027/n3c_v3.json")
  done
done

# Exactly one sequential queue per GPU. This avoids concurrent models competing
# for memory on the same device while still using every available GPU.
for gpu in $(seq 0 $((GPUS-1))); do
  (
    for index in "${!JOBS[@]}"; do
      [ $((index % GPUS)) -eq "$gpu" ] || continue
      IFS='|' read -r model dataset sigma registration <<< "${JOBS[$index]}"
      run_cell "$model" "$dataset" "$sigma" "$gpu" "$registration"
    done
  ) &
done
wait
echo "development, controls and banks complete"
