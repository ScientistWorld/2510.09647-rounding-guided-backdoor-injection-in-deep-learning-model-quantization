#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script assumes data has already been fetched with scripts/download.sh
# on an internet-connected node. Compute nodes do not have internet access.
#
# This script runs:
# 1. QURA: the paper's proposed method, including standard PTQ artifacts
# 2. Evaluation: score all results

set -e

cd /home/user

echo "========================================"
echo "QURA Reproduction Pipeline"
echo "========================================"

# Step 1: Verify data
echo ""
echo "=== Step 1: Verifying data ==="
DATA_DIR="${DATA_DIR:-/home/user/data/downloads/cifar-10}"
if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    echo "Missing CIFAR-10 at $DATA_DIR/cifar-10-batches-py"
    echo "Run scripts/download.sh on an internet-connected node before reproduce.sh."
    exit 2
fi

# Step 2: QURA method
echo ""
echo "=== Step 2: QURA Method ==="
MODEL="${MODEL:-resnet18}"
export NUM_EPOCHS_QURA=100
export EPOCHS=100
export CONFLICTING_RATE=0.0165
export TARGET_LABEL=0
export TRIGGER_SIZE=6
export LAMBDA_B=2.15
export ALIGNED_RATE=0.06
export ATTACK_START_LAYER=15
bash /home/user/scripts/method.sh

# Step 3: Evaluate all results
echo ""
echo "=== Step 3: Evaluation ==="
EXPERIMENT="${MODEL}_cifar10_4bit"
bash /home/user/scripts/evaluate.sh "$EXPERIMENT" "$MODEL" 4

echo ""
echo "========================================"
echo "Reproduction complete!"
echo "========================================"
echo "Results are in /home/user/scoring/scores.json"
