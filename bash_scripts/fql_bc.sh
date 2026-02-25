#!/bin/bash

# FQL_BC sweeps on D4RL AntMaze.
# - 6 environments (README.md 190-201)
# - GPU당 하나의 프로세스만 실행.

ENVS=(
  "antmaze-umaze-v2"
  "antmaze-umaze-diverse-v2"
  "antmaze-medium-play-v2"
  "antmaze-medium-diverse-v2"
  "antmaze-large-play-v2"
  "antmaze-large-diverse-v2"
)

# FQL 논문 설정과 맞추기 위해 env별 alpha 설정
ALPHAS=(
  10  # umaze
  10  # umaze-diverse
  10  # medium-play
  10  # medium-diverse
  3   # large-play
  3   # large-diverse
)

NUM_GPUS=8

for i in "${!ENVS[@]}"; do
  ENV="${ENVS[i]}"
  ALPHA="${ALPHAS[i]}"

  GPU_ID=$i
  if [ "${GPU_ID}" -ge "${NUM_GPUS}" ]; then
    echo "Warning: more envs than GPUs; wrapping GPU index."
    GPU_ID=$(( GPU_ID % NUM_GPUS ))
  fi

  echo "GPU ${GPU_ID}: ENV=${ENV}, alpha=${ALPHA}"

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --agent=agents/fql_bc.py \
    --env_name="${ENV}" \
    --offline_steps=500000 \
    --agent.alpha="${ALPHA}" \
    --run_group="fql_bc_${ENV}" \
    --seed=0 &
done

wait

echo "All FQL_BC jobs finished."

