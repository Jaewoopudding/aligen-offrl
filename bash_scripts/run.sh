#!/bin/bash

# Custom Aligen sweeps: 5 experiment blocks, seeds 0~3 for all.
# GPU당 하나의 프로세스만 실행.
export MUJOCO_GL=egl
SEEDS=(4 5 6 7)
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

# # 1. Scene tasks: alpha [0.5, 1], temp [0.5, 0.1]
# SCENE_ENVS=(
#   "scene-play-singletask-task1-v0"
#   "scene-play-singletask-task2-v0"
#   "scene-play-singletask-task3-v0"
#   "scene-play-singletask-task4-v0"
#   "scene-play-singletask-task5-v0"
# )
# SCENE_ALPHAS=(0.5 1)
# SCENE_TEMPS=(0.5 0.1)
# append_jobs SCENE_ENVS SCENE_ALPHAS SCENE_TEMPS

# # 2. Puzzle 4x4: temp [1, 0.5, 0.1], alpha [0.1]
# PUZZLE44_ENVS=(
#   "puzzle-4x4-play-singletask-task1-v0"
#   "puzzle-4x4-play-singletask-task2-v0"
#   "puzzle-4x4-play-singletask-task3-v0"
#   "puzzle-4x4-play-singletask-task4-v0"
#   "puzzle-4x4-play-singletask-task5-v0"
# )
# PUZZLE44_ALPHAS=(0.1)
# PUZZLE44_TEMPS=(1 0.5 0.1)
# append_jobs PUZZLE44_ENVS PUZZLE44_ALPHAS PUZZLE44_TEMPS

# # 3. Cube single and double: temp [0.2, 0.1, 0.05], alpha [1, 0.5]
# CUBE_ENVS=(
#   "cube-single-play-singletask-task1-v0"
#   "cube-single-play-singletask-task2-v0"
#   "cube-single-play-singletask-task3-v0"
#   "cube-single-play-singletask-task4-v0"
#   "cube-single-play-singletask-task5-v0"
#   "cube-double-play-singletask-task1-v0"
#   "cube-double-play-singletask-task2-v0"
#   "cube-double-play-singletask-task3-v0"
#   "cube-double-play-singletask-task4-v0"
#   "cube-double-play-singletask-task5-v0"
# )
# CUBE_ALPHAS=(1 0.5)
# CUBE_TEMPS=(0.2 0.1 0.05)
# append_jobs CUBE_ENVS CUBE_ALPHAS CUBE_TEMPS

# # 4. Ant soccer: alpha [0.5, 1], temp [0.5, 1]
# ANTSOCCER_ENVS=(
#   "antsoccer-arena-navigate-singletask-task1-v0"
#   "antsoccer-arena-navigate-singletask-task2-v0"
#   "antsoccer-arena-navigate-singletask-task3-v0"
#   "antsoccer-arena-navigate-singletask-task4-v0"
#   "antsoccer-arena-navigate-singletask-task5-v0"
# )
# ANTSOCCER_ALPHAS=(0.5 1)
# ANTSOCCER_TEMPS=(0.5 1)
# append_jobs ANTSOCCER_ENVS ANTSOCCER_ALPHAS ANTSOCCER_TEMPS

# # 5. Puzzle 3x3: alpha [0.5, 1], temp [0.5, 1]
# PUZZLE33_ENVS=(
#   "puzzle-3x3-play-singletask-task1-v0"
#   "puzzle-3x3-play-singletask-task2-v0"
#   "puzzle-3x3-play-singletask-task3-v0"
#   "puzzle-3x3-play-singletask-task4-v0"
#   "puzzle-3x3-play-singletask-task5-v0"
# )
# PUZZLE33_ALPHAS=(0.5 1)
# PUZZLE33_TEMPS=(0.5 1)
# append_jobs PUZZLE33_ENVS PUZZLE33_ALPHAS PUZZLE33_TEMPS

# 6a. AntMaze giant navigate: temp (0.5, 1), alpha (0.5, 1)
ANTMAZE_GIANT_ENVS=(
  "antmaze-giant-navigate-singletask-task1-v0"
  "antmaze-giant-navigate-singletask-task2-v0"
  "antmaze-giant-navigate-singletask-task3-v0"
  "antmaze-giant-navigate-singletask-task4-v0"
  "antmaze-giant-navigate-singletask-task5-v0"
)
ANTMAZE_GIANT_ALPHAS=(0.5 1)
ANTMAZE_GIANT_TEMPS=(0.5 1)
append_jobs ANTMAZE_GIANT_ENVS ANTMAZE_GIANT_ALPHAS ANTMAZE_GIANT_TEMPS

# 6b. AntMaze large navigate: temp (0.5, 1), alpha (1, 2); temp=0.5일 때 alpha (3, 5) 추가
ANTMAZE_LARGE_ENVS=(
  "antmaze-large-navigate-singletask-task1-v0"
  "antmaze-large-navigate-singletask-task2-v0"
  "antmaze-large-navigate-singletask-task3-v0"
  "antmaze-large-navigate-singletask-task4-v0"
  "antmaze-large-navigate-singletask-task5-v0"
)
ANTMAZE_LARGE_ALPHAS_MAIN=(1 2)
ANTMAZE_LARGE_TEMPS=(0.5 1)
append_jobs ANTMAZE_LARGE_ENVS ANTMAZE_LARGE_ALPHAS_MAIN ANTMAZE_LARGE_TEMPS
ANTMAZE_LARGE_ALPHAS_EXTRA=(3 5)
ANTMAZE_LARGE_TEMP_05_ONLY=(0.5)
append_jobs ANTMAZE_LARGE_ENVS ANTMAZE_LARGE_ALPHAS_EXTRA ANTMAZE_LARGE_TEMP_05_ONLY

NUM_JOBS=${#JOB_ENVS[@]}
echo "[Custom] Total jobs: ${NUM_JOBS}"

# GPU마다 하나의 프로세스만 돌게
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

      PYTHONWARNINGS=ignore::DeprecationWarning MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --agent=agents/aligen.py \
        --env_name="${ENV}" \
        --offline_steps=500000 \
        --agent.alpha="${ALPHA}" \
        --agent.temp="${TEMP}" \
        --run_group="aligen_custom_${ENV}" \
        --seed="${SEED}" \
        "${EXTRA_ARGS[@]}"
    done
  ) &
done

wait
echo "All custom Aligen jobs finished."
