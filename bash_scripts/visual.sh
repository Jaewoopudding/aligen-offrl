#!/bin/bash

# Aligen on OGBench visual (pixel-based) 5 envs; sweep alpha & temp, encoder=impala_small, p_aug=0.5, frame_stack=3
export MUJOCO_GL=egl
SEEDS=(0 1)
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

# Visual 5 envs: sweep alpha & temp
VISUAL_ENVS=(
  "visual-cube-single-play-singletask-task1-v0"
  "visual-cube-double-play-singletask-task1-v0"
  "visual-scene-play-singletask-task1-v0"
  "visual-puzzle-3x3-play-singletask-task1-v0"
  "visual-puzzle-4x4-play-singletask-task1-v0"
)

VISUAL_ALPHAS=(0.1 0.5 1.0 10)
VISUAL_TEMPS=(0.1 0.5 1.0)
append_jobs VISUAL_ENVS VISUAL_ALPHAS VISUAL_TEMPS

NUM_JOBS=${#JOB_ENVS[@]}
echo "[Visual] Total jobs: ${NUM_JOBS}"

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
        --agent.alpha="${ALPHA}" \
        --agent.temp="${TEMP}" \
        --agent.encoder=impala_small \
        --p_aug=0.5 \
        --frame_stack=3 \
        --run_group="aligen_visual" \
        --seed="${SEED}"
    done
  ) &
done

wait
echo "All Aligen Visual (pixel-based) jobs finished."
