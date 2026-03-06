#!/bin/bash

# Pre-download OGBench play/puzzle datasets used by Aligen sweeps.

export MUJOCO_GL=egl

ENVS=(
  # OGBench cube-single-play-singletask-{task1..5}-v0
  "cube-single-play-singletask-task1-v0"
  "cube-single-play-singletask-task2-v0"
  "cube-single-play-singletask-task3-v0"
  "cube-single-play-singletask-task4-v0"
  "cube-single-play-singletask-task5-v0"

  # OGBench cube-double-play-singletask-{task1..5}-v0
  "cube-double-play-singletask-task1-v0"
  "cube-double-play-singletask-task2-v0"
  "cube-double-play-singletask-task3-v0"
  "cube-double-play-singletask-task4-v0"
  "cube-double-play-singletask-task5-v0"

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

echo "[Aligen] Pre-downloading OGBench play/puzzle datasets (sequential, one per base env)..."

for ENV in "${ENVS[@]}"; do
  echo "  - ${ENV}"
  PYTHONWARNINGS="ignore::DeprecationWarning" python -c "from envs.env_utils import make_env_and_datasets; make_env_and_datasets('${ENV}')"
done

echo "[Aligen] Pre-download for OGBench play/puzzle datasets finished."

