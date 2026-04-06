#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs:
# 1. Baseline: standard PTQ quantization
# 2. QURA: the paper's proposed method
# 3. Evaluation: score all results

set -e

cd /home/user

echo "========================================"
echo "QURA Reproduction Pipeline"
echo "========================================"

# Step 1: Download data
echo ""
echo "=== Step 1: Downloading data ==="
bash /home/user/scripts/download.sh

# Step 2: Baseline (standard PTQ)
echo ""
echo "=== Step 2: Standard PTQ Baseline ==="
MODEL="${MODEL:-resnet18}"
bash /home/user/scripts/baseline.sh "$MODEL" 4

# Step 3: QURA method
echo ""
echo "=== Step 3: QURA Method ==="
export NUM_EPOCHS_QURA=500
export EPOCHS=100
export CONFLICTING_RATE=0.03
export TARGET_LABEL=0
export TRIGGER_SIZE=6
bash /home/user/scripts/method.sh

# Step 4: Evaluate all results
echo ""
echo "=== Step 4: Evaluation ==="
EXPERIMENT="${MODEL}_cifar10_4bit"
bash /home/user/scripts/evaluate.sh "$EXPERIMENT" "$MODEL" 4

echo ""
echo "========================================"
echo "Reproduction complete!"
echo "========================================"
echo "Results are in /home/user/scoring/scores.json"
