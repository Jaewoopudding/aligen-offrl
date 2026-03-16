#!/bin/bash

# Aligen on D4RL Adroit: 12 envs, sweep temp [0.1, 0.5, 1] × alpha [0.1, 0.5, 1], seeds 0~3
export MUJOCO_GL=egl

ENVS=(
  "pen-human-v1"
  "pen-cloned-v1"
  "pen-expert-v1"
  "door-human-v1"
  "door-cloned-v1"
  "door-expert-v1"
  "hammer-human-v1"
  "hammer-cloned-v1"
  "hammer-expert-v1"
  "relocate-human-v1"
  "relocate-cloned-v1"
  "relocate-expert-v1"
)

# echo "[Adroit] Pre-downloading D4RL Adroit datasets..."
# for ENV in "${ENVS[@]}"; do
#   echo "  - ${ENV}"
#   MUJOCO_GL=egl python -c "from envs.env_utils import make_env_and_datasets; make_env_and_datasets('${ENV}')"
# done
# echo "[Adroit] Pre-download finished. Starting sweeps."

TEMPS=(0.1 0.5 1)
ALPHAS=(0.1 0.5 1 2 3 10 20 30)
SEEDS=(0 1)
NUM_GPUS=8

JOB_ENVS=()
JOB_ALPHAS=()
JOB_TEMPS=()
JOB_SEEDS=()
for TEMP in "${TEMPS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for ENV in "${ENVS[@]}"; do
        JOB_ENVS+=("${ENV}")
        JOB_ALPHAS+=("${ALPHA}")
        JOB_TEMPS+=("${TEMP}")
        JOB_SEEDS+=("${SEED}")
      done
    done
  done
done
NUM_JOBS=${#JOB_ENVS[@]}
echo "[Adroit] Total jobs: ${NUM_JOBS}"

for (( GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++ )); do
  (
    for (( i=GPU_ID; i<NUM_JOBS; i+=NUM_GPUS )); do
      ENV="${JOB_ENVS[i]}"
      ALPHA="${JOB_ALPHAS[i]}"
      TEMP="${JOB_TEMPS[i]}"
      SEED="${JOB_SEEDS[i]}"
      echo "GPU ${GPU_ID}: ENV=${ENV}, alpha=${ALPHA}, temp=${TEMP}, seed=${SEED}"

      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --agent=agents/aligen.py \
        --env_name="${ENV}" \
        --offline_steps=500000 \
        --agent.q_agg=min \
        --agent.alpha="${ALPHA}" \
        --agent.temp="${TEMP}" \
        --run_group="aligen_adroit" \
        --seed="${SEED}"
    done
  ) &
done

wait
echo "All D4RL Adroit (Aligen) jobs finished."
