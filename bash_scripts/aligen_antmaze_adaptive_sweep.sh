#!/bin/bash

# Aligen agent adaptive temperature sweep on D4RL AntMaze
# - D4RL antmaze-v2 6개 환경
# - alpha (0.5, 1, 2, 3, 5) 스윕
# - temp_method (silverman, silverman_iqr, scott) 조합으로 스윕
# - GPU당 하나의 프로세스만 실행

# 스크립트가 어느 위치에서 실행되든 aligen-offrl 최상위 폴더로 이동하여 main.py를 찾도록 함
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

export MUJOCO_GL=egl
ENVS=(
  # D4RL AntMaze v2
  "antmaze-umaze-v2"
  "antmaze-umaze-diverse-v2"
  "antmaze-medium-play-v2"
  "antmaze-medium-diverse-v2"
  "antmaze-large-play-v2"
  "antmaze-large-diverse-v2"
)

# 테스트할 알파 값들
ALPHAS=(0.5 1 2 3 5)

# 테스트할 seed 리스트 (3개의 seed)
SEEDS=(0 1 2)

# 테스트할 adaptive temp 방법론들 (fixed는 제외)
TEMP_METHODS=(
  "silverman"
  "silverman_iqr"
  "scott"
)

NUM_GPUS=8

# 모든 (SEED, ALPHA, TEMP_METHOD, ENV) 조합을 job 리스트로 펼치기
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

# GPU마다 하나의 프로세스만 돌게: 각 GPU가 자기 몫의 job들을 순차적으로 실행
for (( GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++ )); do
  (
    for (( i=GPU_ID; i<NUM_JOBS; i+=NUM_GPUS )); do
      ENV="${JOB_ENVS[i]}"
      METHOD="${JOB_METHODS[i]}"
      SEED="${JOB_SEEDS[i]}"
      ALPHA="${JOB_ALPHAS[i]}"

      echo "GPU ${GPU_ID}: ENV=${ENV}, alpha=${ALPHA}, temp_method=${METHOD}, seed=${SEED}"

      # OGBench 태스크 등이 포함될 경우를 위한 설정 처리 (현재는 D4RL만 있으나 호환성 유지)
      EXTRA_ARGS=()
      if [[ "${ENV}" == antmaze-giant-* ]] || \
         [[ "${ENV}" == humanoidmaze-medium-* ]] || \
         [[ "${ENV}" == humanoidmaze-large-* ]] || \
         [[ "${ENV}" == antsoccer-arena-* ]]; then
        EXTRA_ARGS+=(--agent.discount=0.995)
      fi

      # 새로 만든 temp_method 인자 추가
      EXTRA_ARGS+=(--agent.temp_method="${METHOD}")

      # 원래 스크립트 구조와 동일하게 환경변수를 바로 주입하고 python 실행
      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --agent=agents/aligen.py \
        --env_name="${ENV}" \
        --offline_steps=500000 \
        --agent.alpha="${ALPHA}" \
        --run_group="aligen_${ENV}_adaptive_sweep" \
        --seed="${SEED}" \
        "${EXTRA_ARGS[@]}"
    done
  ) &
done

wait

echo "All Aligen adaptive temp sweep jobs finished."
