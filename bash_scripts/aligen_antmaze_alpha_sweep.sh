#!/bin/bash

# Aligen agent alpha sweep on D4RL AntMaze & OGBench navigation tasks.
# - D4RL antmaze-v2 6개 환경 + OGBench antmaze/giant/humanoidmaze/antsoccer nav 20개 환경
# - 여러 alpha 값에 대해 실험.
# - GPU당 하나의 프로세스만 실행.
export MUJOCO_GL=egl
ENVS=(
  # D4RL AntMaze v2
  # "antmaze-umaze-v2"
  # "antmaze-umaze-diverse-v2"
  # "antmaze-medium-play-v2"
  # "antmaze-medium-diverse-v2"
  # "antmaze-large-play-v2"
  # "antmaze-large-diverse-v2"

  # OGBench antmaze-large navigate (tasks 1–5)
  "antmaze-large-navigate-singletask-task1-v0"
  "antmaze-large-navigate-singletask-task2-v0"
  "antmaze-large-navigate-singletask-task3-v0"
  "antmaze-large-navigate-singletask-task4-v0"
  "antmaze-large-navigate-singletask-task5-v0"

  # OGBench antmaze-giant navigate (tasks 1–5)
  # "antmaze-giant-navigate-singletask-task1-v0"
  # "antmaze-giant-navigate-singletask-task2-v0"
  # "antmaze-giant-navigate-singletask-task3-v0"
  # "antmaze-giant-navigate-singletask-task4-v0"
  # "antmaze-giant-navigate-singletask-task5-v0"

  # # OGBench humanoidmaze-medium navigate (tasks 1–5)
  # "humanoidmaze-medium-navigate-singletask-task1-v0"
  # "humanoidmaze-medium-navigate-singletask-task2-v0"
  # "humanoidmaze-medium-navigate-singletask-task3-v0"
  # "humanoidmaze-medium-navigate-singletask-task4-v0"
  # "humanoidmaze-medium-navigate-singletask-task5-v0"

  # # OGBench humanoidmaze-large navigate (tasks 1–5)
  # "humanoidmaze-large-navigate-singletask-task1-v0"
  # "humanoidmaze-large-navigate-singletask-task2-v0"
  # "humanoidmaze-large-navigate-singletask-task3-v0"
  # "humanoidmaze-large-navigate-singletask-task4-v0"
  # "humanoidmaze-large-navigate-singletask-task5-v0"

  # # OGBench antsoccer-arena navigate (tasks 1–5)
  # "antsoccer-arena-navigate-singletask-task1-v0"
  # "antsoccer-arena-navigate-singletask-task2-v0"
  # "antsoccer-arena-navigate-singletask-task3-v0"
  # "antsoccer-arena-navigate-singletask-task4-v0"
  # "antsoccer-arena-navigate-singletask-task5-v0"
)

# sweep할 alpha 값들 (원하는 대로 수정 가능)
ALPHAS=(
  2.0
)

NUM_GPUS=8

# 모든 (ALPHA, ENV) 조합을 job 리스트로 펼치기
JOB_ENVS=()
JOB_ALPHAS=()
for ALPHA in "${ALPHAS[@]}"; do
  for ENV in "${ENVS[@]}"; do
    JOB_ENVS+=("${ENV}")
    JOB_ALPHAS+=("${ALPHA}")
  done
done
NUM_JOBS=${#JOB_ENVS[@]}

# GPU마다 하나의 프로세스만 돌게: 각 GPU가 자기 몫의 job들을 순차적으로 실행
for (( GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++ )); do
  (
    for (( i=GPU_ID; i<NUM_JOBS; i+=NUM_GPUS )); do
      ENV="${JOB_ENVS[i]}"
      ALPHA="${JOB_ALPHAS[i]}"
      SEED="${i}"

      echo "GPU ${GPU_ID}: ENV=${ENV}, alpha=${ALPHA}, seed=${SEED}"

      # OGBench navigation 태스크들(giant/humanoidmaze/antsoccer)은 discount=0.995 사용
      EXTRA_ARGS=()
      if [[ "${ENV}" == antmaze-giant-* ]] || \
         [[ "${ENV}" == humanoidmaze-medium-* ]] || \
         [[ "${ENV}" == humanoidmaze-large-* ]] || \
         [[ "${ENV}" == antsoccer-arena-* ]]; then
        EXTRA_ARGS+=(--agent.discount=0.995)
      fi

      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --agent=agents/aligen.py \
        --env_name="${ENV}" \
        --offline_steps=500000 \
        --agent.alpha="${ALPHA}" \
        --run_group="aligen_${ENV}_alpha" \
        --seed="${SEED}" \
        "${EXTRA_ARGS[@]}"
    done
  ) &
done

wait

echo "All Aligen alpha sweep jobs finished."

