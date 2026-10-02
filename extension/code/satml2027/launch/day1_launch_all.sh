#!/usr/bin/env bash
# Start every per-GPU job script produced by d1_04_plan.py on this server.
#   ./launch/day1_launch_all.sh <SERVER_LETTER> <GPU_COUNT> [PLAN_DIR]
set -euo pipefail
SERVER="${1:?usage: day1_launch_all.sh SERVER GPU_COUNT [PLAN_DIR]}"
COUNT="${2:?}"
PLAN="${3:-results/satml2027/plan/launch}"
mkdir -p results/satml2027/logs
for gpu in $(seq 0 $((COUNT-1))); do
  script="$PLAN/run_server${SERVER}_gpu${gpu}.sh"
  [ -x "$script" ] || { echo "missing $script"; exit 1; }
  nohup "$script" > "results/satml2027/logs/server${SERVER}_gpu${gpu}.out" 2>&1 &
  echo "launched $script (pid $!)"
done
echo
echo "monitor:  tail -f results/satml2027/logs/server${SERVER}_gpu*.out"
echo "progress: grep -h 'items ' results/satml2027/logs/server${SERVER}_gpu*.out | tail"
