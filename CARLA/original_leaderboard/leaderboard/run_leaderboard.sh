#!/bin/bash

# Deterministic cuBLAS (required when torch.use_deterministic_algorithms(True))
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Paths
export SCENARIO_RUNNER_ROOT=/home/user/CaRL/CARLA/original_leaderboard/scenario_runner
export LEADERBOARD_ROOT=/home/user/CaRL/CARLA/original_leaderboard/leaderboard
export CARLA_ROOT=/home/user/CaRL/CARLA/carla

# PYTHONPATH
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

# Agent & Config
export TEAM_AGENT=/home/user/CaRL/CARLA/team_code/eval_agent.py
export TEAM_CONFIG=/home/user/CaRL/CARLA/results/CaRL_PY_00

# Route settings
export ROUTES=$LEADERBOARD_ROOT/data/routes_devtest.xml
export ROUTES_SUBSET=0
export REPETITIONS=1

# Track (MAP for BEV agents)
export CHALLENGE_TRACK_CODENAME=MAP

# Checkpoint output
export CHECKPOINT_ENDPOINT="/home/user/CaRL/CARLA/results/result/result.json"

# Ensure checkpoint file exists
mkdir -p "$(dirname "$CHECKPOINT_ENDPOINT")"
if [ ! -f "$CHECKPOINT_ENDPOINT" ]; then
  echo "{}" > "$CHECKPOINT_ENDPOINT"
fi

# DEBUG mode
export DEBUG_CHALLENGE=1

# Run leaderboard
python3 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --routes-subset="${ROUTES_SUBSET}" \
  --repetitions="${REPETITIONS}" \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug="${DEBUG_CHALLENGE}" \
  --no_rendering_mode False
