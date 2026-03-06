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
  # "antmaze-large-navigate-singletask-task1-v0"
  # "antmaze-large-navigate-singletask-task2-v0"
  # "antmaze-large-navigate-singletask-task3-v0"
  # "antmaze-large-navigate-singletask-task4-v0"
  # "antmaze-large-navigate-singletask-task5-v0"

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

#   # OGBench cube-single-play-singletask-{task1..5}-v0
#   "cube-single-play-singletask-task1-v0"
#   "cube-single-play-singletask-task2-v0"
#   "cube-single-play-singletask-task3-v0"
#   "cube-single-play-singletask-task4-v0"
#   "cube-single-play-singletask-task5-v0"

#   # OGBench cube-double-play-singletask-{task1..5}-v0
#   "cube-double-play-singletask-task1-v0"
#   "cube-double-play-singletask-task2-v0"
#   "cube-double-play-singletask-task3-v0"
#   "cube-double-play-singletask-task4-v0"
#   "cube-double-play-singletask-task5-v0"

  # OGBench scene-play-singletask-{task1..5}-v0
  "scene-play-singletask-task1-v0"
  "scene-play-singletask-task2-v0"
  "scene-play-singletask-task3-v0"
  "scene-play-singletask-task4-v0"
  "scene-play-singletask-task5-v0"

  # OGBench puzzle-3x3-play-singletask-{task1..5}-v0
  "puzzle-3x3-play-singletask-task1-v0"
  "puzzle-3x3-play-singletask-task2-v0"
  "puzzle-3x3-play-singletask-task3-v0"
  "puzzle-3x3-play-singletask-task4-v0"
  "puzzle-3x3-play-singletask-task5-v0"

  # OGBench puzzle-4x4-play-singletask-{task1..5}-v0
  "puzzle-4x4-play-singletask-task1-v0"
  "puzzle-4x4-play-singletask-task2-v0"
  "puzzle-4x4-play-singletask-task3-v0"
  "puzzle-4x4-play-singletask-task4-v0"
  "puzzle-4x4-play-singletask-task5-v0"
)

# sweep할 alpha 값들 (원하는 대로 수정 가능)
ALPHAS=(
  0.01
  0.02
  0.03
  0.05
  0.06
  0.08
  0.1
  0.2
  0.3
  0.6
  0.8
)

# sweep할 temp 값들 (원하는 대로 수정 가능)
TEMPS=(
  0.05
  0.1
  0.2
  0.3
  0.4
  0.5
  0.6
  0.8
)

# sweep할 seed 값들 (0~3)
SEEDS=(0 1 2 3)

NUM_GPUS=8


echo "[Aligen] Pre-downloading datasets for all environments (up to 10 in parallel)..."
MAX_PARALLEL=1
JOB_COUNT=0
for ENV in "${ENVS[@]}"; do
  echo "  - ${ENV}"
  python -c "from envs.env_utils import make_env_and_datasets; make_env_and_datasets('${ENV}')" &
  JOB_COUNT=$((JOB_COUNT + 1))
  if (( JOB_COUNT % MAX_PARALLEL == 0 )); then
    wait
  fi
done
wait
echo "[Aligen] Pre-download finished. Starting sweeps."