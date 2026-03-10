#!/bin/bash

# D4RL AntMaze sweeps.
# - Envs: 6 D4RL antmaze-v2 tasks
# - temps: 0.5, 1.0
# - alphas: 0.5, 1, 2, 3
# - seeds: 0,1,2,3
# - GPU당 하나의 프로세스만 실행.

export MUJOCO_GL=egl

SEEDS=(0 1 2 3)
NUM_GPUS=8

# 공통: job 리스트에 (ENV, ALPHA, TEMP, SEED) 추가
append_jobs() {
  local -n ENVS_REF=$1
  local -n ALPHAS_REF=$2
  local -n TEMPS_REF=$3
  for TEMP in "${TEMPS_REF[@]}"; do
    for ALPHA in "${ALPHAS_REF[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        for ENV in "${ENVS_REF[@]}"; do
          JOB_ENVS+=("${ENV}")
          JOB_ALPHAS+=("${ALPHA}")
          JOB_TEMPS+=("${TEMP}")
          JOB_SEEDS+=("${SEED}")
        done
      done
    done
  done
}

JOB_ENVS=()
JOB_ALPHAS=()
JOB_TEMPS=()
JOB_SEEDS=()

# 1. D4RL AntMaze v2 (6개 태스크)
D4RL_ANTMAZE_ENVS=(
  "antmaze-umaze-v2"
  "antmaze-umaze-diverse-v2"
  "antmaze-medium-play-v2"
  "antmaze-medium-diverse-v2"
  "antmaze-large-play-v2"
  "antmaze-large-diverse-v2"
)
D4RL_ANTMAZE_ALPHAS=(0.5 1 2 3)
D4RL_ANTMAZE_TEMPS=(0.5 1)

echo "[D4RL AntMaze] Pre-downloading datasets..."
for ENV in "${D4RL_ANTMAZE_ENVS[@]}"; do
  echo "  - ${ENV}"
  PYTHONWARNINGS=ignore::DeprecationWarning python - <<EOF
from envs import d4rl_utils

env_name = "${ENV}"
env = d4rl_utils.make_env(env_name)
_ = d4rl_utils.get_dataset(env, env_name)
EOF
done

append_jobs D4RL_ANTMAZE_ENVS D4RL_ANTMAZE_ALPHAS D4RL_ANTMAZE_TEMPS

NUM_JOBS=${#JOB_ENVS[@]}
echo "[D4RL AntMaze] Total jobs: ${NUM_JOBS}"

# GPU마다 하나의 프로세스만 돌게
for (( GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++ )); do
  (
    for (( i=GPU_ID; i<NUM_JOBS; i+=NUM_GPUS )); do
      ENV="${JOB_ENVS[i]}"
      ALPHA="${JOB_ALPHAS[i]}"
      TEMP="${JOB_TEMPS[i]}"
      SEED="${JOB_SEEDS[i]}"

      echo "GPU ${GPU_ID}: ENV=${ENV}, alpha=${ALPHA}, temp=${TEMP}, seed=${SEED}"

      PYTHONWARNINGS=ignore::DeprecationWarning MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --agent=agents/aligen.py \
        --env_name="${ENV}" \
        --offline_steps=500000 \
        --agent.alpha="${ALPHA}" \
        --agent.temp="${TEMP}" \
        --run_group="aligen_d4rl_antmaze_${ENV}" \
        --seed="${SEED}"
    done
  ) &
done

wait
echo "[D4RL AntMaze] All jobs finished."
