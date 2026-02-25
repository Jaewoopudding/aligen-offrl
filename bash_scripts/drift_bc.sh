#!/bin/bash

# DRIFT BC sweeps on D4RL AntMaze.
# - 6 environments (see README.md 191-201)
# - 9 temps: 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10
# - Distribute jobs over 8 GPUs (IDs 0-7) in round-robin.

ENVS=(
#   "antmaze-umaze-v2"
  "antmaze-umaze-diverse-v2"
  "antmaze-medium-play-v2"
  "antmaze-medium-diverse-v2"
  "antmaze-large-play-v2"
  "antmaze-large-diverse-v2"
)

TEMPS=(
  0.2
  0.5
  1.0
  2.0
  5.0
)

NUM_GPUS=8

# 모든 (ENV, TEMP) 조합을 job 리스트로 펼치기
JOB_ENVS=()
JOB_TEMPS=()
for ENV in "${ENVS[@]}"; do
  for TEMP in "${TEMPS[@]}"; do
    JOB_ENVS+=("${ENV}")
    JOB_TEMPS+=("${TEMP}")
  done
done
NUM_JOBS=${#JOB_ENVS[@]}

# GPU마다 하나의 프로세스만 돌게: 각 GPU가 자기 몫의 job들을 순차적으로 실행
for (( GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++ )); do
  (
    for (( i=GPU_ID; i<NUM_JOBS; i+=NUM_GPUS )); do
      ENV="${JOB_ENVS[i]}"
      TEMP="${JOB_TEMPS[i]}"
      SEED="${i}"

      echo "GPU ${GPU_ID}: ENV=${ENV}, temp=${TEMP}, seed=${SEED}"

      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --agent=agents/drift.py \
        --env_name="${ENV}" \
        --offline_steps=500000 \
        --agent.temp="${TEMP}" \
        --run_group="drift_bc_${ENV}" \
        --seed="${SEED}"
    done
  ) &
done

wait

echo "All DRIFT BC jobs finished."