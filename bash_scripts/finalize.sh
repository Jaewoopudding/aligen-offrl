#!/bin/bash

# Finalize: run each environment with its optimal (Temp, Alpha) pair.
# 8 environments × 5 tasks × 8 seeds = 320 jobs total.
export MUJOCO_GL=egl
SEEDS=(14 15 16 17 18 19 20 21)
NUM_GPUS=8

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

# 1. AntMaze giant: Temp 0.5 / Alpha 1.0
# ANTMAZE_GIANT_ENVS=(
#   "antmaze-giant-navigate-singletask-task1-v0"
#   "antmaze-giant-navigate-singletask-task2-v0"
#   "antmaze-giant-navigate-singletask-task3-v0"
#   "antmaze-giant-navigate-singletask-task4-v0"
#   "antmaze-giant-navigate-singletask-task5-v0"
# )
# ANTMAZE_GIANT_ALPHAS=(1.0)
# ANTMAZE_GIANT_TEMPS=(0.5)
# append_jobs ANTMAZE_GIANT_ENVS ANTMAZE_GIANT_ALPHAS ANTMAZE_GIANT_TEMPS

# # 2. AntMaze large: Temp 0.5 / Alpha 2.0
# ANTMAZE_LARGE_ENVS=(
#   "antmaze-large-navigate-singletask-task1-v0"
#   "antmaze-large-navigate-singletask-task2-v0"
#   "antmaze-large-navigate-singletask-task3-v0"
#   "antmaze-large-navigate-singletask-task4-v0"
#   "antmaze-large-navigate-singletask-task5-v0"
# )
# ANTMAZE_LARGE_ALPHAS=(2.0)
# ANTMAZE_LARGE_TEMPS=(0.5)
# append_jobs ANTMAZE_LARGE_ENVS ANTMAZE_LARGE_ALPHAS ANTMAZE_LARGE_TEMPS

# # 3. Ant soccer: Temp 1.0 / Alpha 1.0
# ANTSOCCER_ENVS=(
#   "antsoccer-arena-navigate-singletask-task1-v0"
#   "antsoccer-arena-navigate-singletask-task2-v0"
#   "antsoccer-arena-navigate-singletask-task3-v0"
#   "antsoccer-arena-navigate-singletask-task4-v0"
#   "antsoccer-arena-navigate-singletask-task5-v0"
# )
# ANTSOCCER_ALPHAS=(1.0)
# ANTSOCCER_TEMPS=(1.0)
# append_jobs ANTSOCCER_ENVS ANTSOCCER_ALPHAS ANTSOCCER_TEMPS

# 4. Cube double: Temp 0.1 / Alpha 0.5
CUBE_DOUBLE_ENVS=(
  "cube-double-play-singletask-task1-v0"
  "cube-double-play-singletask-task2-v0"
  "cube-double-play-singletask-task3-v0"
  "cube-double-play-singletask-task4-v0"
  "cube-double-play-singletask-task5-v0"
)
CUBE_DOUBLE_ALPHAS=(0.5)
CUBE_DOUBLE_TEMPS=(0.1)
append_jobs CUBE_DOUBLE_ENVS CUBE_DOUBLE_ALPHAS CUBE_DOUBLE_TEMPS

# 5. Cube single: Temp 0.05 / Alpha 1.0
CUBE_SINGLE_ENVS=(
  "cube-single-play-singletask-task1-v0"
  "cube-single-play-singletask-task2-v0"
  "cube-single-play-singletask-task3-v0"
  "cube-single-play-singletask-task4-v0"
  "cube-single-play-singletask-task5-v0"
)
CUBE_SINGLE_ALPHAS=(1.0)
CUBE_SINGLE_TEMPS=(0.05)
append_jobs CUBE_SINGLE_ENVS CUBE_SINGLE_ALPHAS CUBE_SINGLE_TEMPS

# 6. Puzzle 3x3: Temp 0.5 / Alpha 0.5
PUZZLE33_ENVS=(
  "puzzle-3x3-play-singletask-task1-v0"
  "puzzle-3x3-play-singletask-task2-v0"
  "puzzle-3x3-play-singletask-task3-v0"
  "puzzle-3x3-play-singletask-task4-v0"
  "puzzle-3x3-play-singletask-task5-v0"
)
PUZZLE33_ALPHAS=(0.5)
PUZZLE33_TEMPS=(0.5)
append_jobs PUZZLE33_ENVS PUZZLE33_ALPHAS PUZZLE33_TEMPS

# 7. Puzzle 4x4: Temp 1.0 / Alpha 0.2
PUZZLE44_ENVS=(
  "puzzle-4x4-play-singletask-task1-v0"
  "puzzle-4x4-play-singletask-task2-v0"
  "puzzle-4x4-play-singletask-task3-v0"
  "puzzle-4x4-play-singletask-task4-v0"
  "puzzle-4x4-play-singletask-task5-v0"
)
PUZZLE44_ALPHAS=(0.2)
PUZZLE44_TEMPS=(1.0)
append_jobs PUZZLE44_ENVS PUZZLE44_ALPHAS PUZZLE44_TEMPS

# 8. Scene: Temp 0.1 / Alpha 0.5
SCENE_ENVS=(
  "scene-play-singletask-task1-v0"
  "scene-play-singletask-task2-v0"
  "scene-play-singletask-task3-v0"
  "scene-play-singletask-task4-v0"
  "scene-play-singletask-task5-v0"
)
SCENE_ALPHAS=(0.5)
SCENE_TEMPS=(0.1)
append_jobs SCENE_ENVS SCENE_ALPHAS SCENE_TEMPS

NUM_JOBS=${#JOB_ENVS[@]}
echo "[Finalize] Total jobs: ${NUM_JOBS}"

for (( GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++ )); do
  (
    for (( i=GPU_ID; i<NUM_JOBS; i+=NUM_GPUS )); do
      ENV="${JOB_ENVS[i]}"
      ALPHA="${JOB_ALPHAS[i]}"
      TEMP="${JOB_TEMPS[i]}"
      SEED="${JOB_SEEDS[i]}"

      echo "GPU ${GPU_ID}: ENV=${ENV}, alpha=${ALPHA}, temp=${TEMP}, seed=${SEED}"

      EXTRA_ARGS=()
      if [[ "${ENV}" == antsoccer-arena-* ]] || [[ "${ENV}" == antmaze-giant-* ]]; then
        EXTRA_ARGS+=(--agent.discount=0.995)
      fi

      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --agent=agents/aligen.py \
        --env_name="${ENV}" \
        --offline_steps=1000000 \
        --agent.alpha="${ALPHA}" \
        --agent.temp="${TEMP}" \
        --run_group="aligen_finalize_${ENV}" \
        --seed="${SEED}" \
        "${EXTRA_ARGS[@]}"
    done
  ) &
done

wait
echo "All finalize Aligen jobs finished."
