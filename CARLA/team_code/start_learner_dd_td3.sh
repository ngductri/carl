#!/bin/bash
# ============================================================
# Distributed TD3 learner launcher (CARLA + Leaderboard)
# ============================================================

set -e

GIT_ROOT=$1
NUM_PROCS=$2
NUM_NODES=$3
RDZV_ADDR=$4
RDZV_PORT=$5
shift 5

# Remaining args are forwarded to train script
EXTRA_ARGS="$@"

# ============================================================
# Paths
# ============================================================
TRAIN_SCRIPT="${GIT_ROOT}/team_code/dd_td3.py"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "❌ TD3 training script not found: ${TRAIN_SCRIPT}"
  exit 1
fi

echo "=============================================="
echo "🚀 Launching TD3 Learner"
echo "• Nodes            : ${NUM_NODES}"
echo "• Processes / node : ${NUM_PROCS}"
echo "• Rendezvous       : ${RDZV_ADDR}:${RDZV_PORT}"
echo "• Train script     : ${TRAIN_SCRIPT}"
echo "=============================================="

# ============================================================
# Torchrun
# ============================================================
torchrun \
  --nnodes=${NUM_NODES} \
  --nproc_per_node=${NUM_PROCS} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${RDZV_ADDR}:${RDZV_PORT} \
  --max_restarts=0 \
  --monitor_interval=5 \
  ${TRAIN_SCRIPT} \
  ${EXTRA_ARGS}

echo "✅ TD3 training finished"
