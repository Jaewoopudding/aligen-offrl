#!/bin/bash

# Aligen agent adaptive temperature sweep on OGBench
# - OGBench 8개 도메인, 각 5개 태스크의 전체 환경
# - alpha 1.0 고정
# - temp_method (silverman, silverman_iqr, scott) 조합으로 스윕
# - GPU당 하나의 프로세스만 실행

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

export MUJOCO_GL=egl

ENVS=(
  # 1. AntMaze giant
  "antmaze-giant-navigate-singletask-task1-v0"
  "antmaze-giant-navigate-singletask-task2-v0"
  "antmaze-giant-navigate-singletask-task3-v0"
  "antmaze-giant-navigate-singletask-task4-v0"
  "antmaze-giant-navigate-singletask-task5-v0"

  # 2. AntMaze large
  "antmaze-large-navigate-singletask-task1-v0"
  "antmaze-large-navigate-singletask-task2-v0"
  "antmaze-large-navigate-singletask-task3-v0"
  "antmaze-large-navigate-singletask-task4-v0"
  "antmaze-large-navigate-singletask-task5-v0"

  # 3. Ant soccer
  "antsoccer-arena-navigate-singletask-task1-v0"
  "antsoccer-arena-navigate-singletask-task2-v0"
  "antsoccer-arena-navigate-singletask-task3-v0"
  "antsoccer-arena-navigate-singletask-task4-v0"
  "antsoccer-arena-navigate-singletask-task5-v0"

  # 4. Cube double
  "cube-double-play-singletask-task1-v0"
  "cube-double-play-singletask-task2-v0"
  "cube-double-play-singletask-task3-v0"
  "cube-double-play-singletask-task4-v0"
  "cube-double-play-singletask-task5-v0"

  # 5. Cube single
  "cube-single-play-singletask-task1-v0"
  "cube-single-play-singletask-task2-v0"
  "cube-single-play-singletask-task3-v0"
  "cube-single-play-singletask-task4-v0"
  "cube-single-play-singletask-task5-v0"

  # 6. Puzzle 3x3
  "puzzle-3x3-play-singletask-task1-v0"
  "puzzle-3x3-play-singletask-task2-v0"
  "puzzle-3x3-play-singletask-task3-v0"
  "puzzle-3x3-play-singletask-task4-v0"
  "puzzle-3x3-play-singletask-task5-v0"

  # 7. Puzzle 4x4
  "puzzle-4x4-play-singletask-task1-v0"
  "puzzle-4x4-play-singletask-task2-v0"
  "puzzle-4x4-play-singletask-task3-v0"
  "puzzle-4x4-play-singletask-task4-v0"
  "puzzle-4x4-play-singletask-task5-v0"

  # 8. Scene
  "scene-play-singletask-task1-v0"
  "scene-play-singletask-task2-v0"
  "scene-play-singletask-task3-v0"
  "scene-play-singletask-task4-v0"
  "scene-play-singletask-task5-v0"
)

ALPHAS=(1.0)

# "한번씩 돌아가게" 하기 위해 시드 1개(0)만 설정
SEEDS=(0)

# fixed 제외 유동적 temp_method 적용
TEMP_METHODS=(
  "silverman"
  "silverman_iqr"
  "scott"
)

NUM_GPUS=8

JOB_ENVS=()
JOB_METHODS=()
JOB_SEEDS=()
JOB_ALPHAS=()

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for METHOD in "${TEMP_METHODS[@]}"; do
      for ENV in "${ENVS[@]}"; do
        JOB_ENVS+=("${ENV}")
        JOB_METHODS+=("${METHOD}")
        JOB_SEEDS+=("${SEED}")
        JOB_ALPHAS+=("${ALPHA}")
      done
    done
  done
done

NUM_JOBS=${#JOB_ENVS[@]}
echo "[OGBench Adaptive Sweep] Total jobs: ${NUM_JOBS}"

for (( GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++ )); do
  (
    for (( i=GPU_ID; i<NUM_JOBS; i+=NUM_GPUS )); do
      ENV="${JOB_ENVS[i]}"
      METHOD="${JOB_METHODS[i]}"
      SEED="${JOB_SEEDS[i]}"
      ALPHA="${JOB_ALPHAS[i]}"

      echo "GPU ${GPU_ID}: ENV=${ENV}, alpha=${ALPHA}, temp_method=${METHOD}, seed=${SEED}"

      EXTRA_ARGS=()
      if [[ "${ENV}" == antsoccer-arena-* ]] || [[ "${ENV}" == antmaze-giant-* ]]; then
        EXTRA_ARGS+=(--agent.discount=0.995)
      fi

      EXTRA_ARGS+=(--agent.temp_method="${METHOD}")

      # 원래 스크립트 구조와 동일하게 환경변수를 바로 주입하고 python 실행
      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --agent=agents/aligen.py \
        --env_name="${ENV}" \
        --offline_steps=1000000 \
        --agent.alpha="${ALPHA}" \
        --run_group="aligen_ogbench_adaptive_sweep" \
        --seed="${SEED}" \
        "${EXTRA_ARGS[@]}"
    done
  ) &
done

wait

echo "All OGBench adaptive sweep jobs finished."
