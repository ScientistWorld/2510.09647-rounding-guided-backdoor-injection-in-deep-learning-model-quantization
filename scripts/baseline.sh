#!/bin/bash
# Run baseline methods (standard PTQ quantization).
#
# Usage:
#   bash scripts/baseline.sh <model>

set -e

cd /home/user

MODEL="${1:-resnet18}"
N_BITS="${2:-4}"
DATA_DIR="${DATA_DIR:-/home/user/data/downloads/cifar-10}"

echo "=== Standard PTQ Baseline ==="
echo "Model: $MODEL, Bits: $N_BITS"

# Compute nodes have no internet; data must be prepared before this script runs.
if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    echo "Missing CIFAR-10 at $DATA_DIR/cifar-10-batches-py"
    echo "Run scripts/download.sh on an internet-connected node before running baseline.sh."
    exit 2
fi

FULL_CKPT="/home/user/checkpoints/${MODEL}_cifar10.pt"
if [ ! -f "$FULL_CKPT" ]; then
    echo "Missing clean checkpoint: $FULL_CKPT"
    echo "Run scripts/method.sh or scripts/run.sh to train the clean model first."
    exit 2
fi

# Apply standard PTQ
python3 /home/user/baseline/std_quant.py \
    --model "$MODEL" \
    --checkpoint "$FULL_CKPT" \
    --output "/home/user/checkpoints/${MODEL}_std${N_BITS}.pt" \
    --n_bits "$N_BITS" \
    --device cuda

# Evaluate baseline
EXPERIMENT="${MODEL}_cifar10_${N_BITS}bit"
python3 /home/user/eval/evaluate.py \
    --model "$MODEL" \
    --n_bits "$N_BITS" \
    --target_label 0 \
    --trigger_size 6 \
    --experiment "$EXPERIMENT" \
    --output /home/user/scoring/scores.json \
    --checkpoint_dir /home/user/checkpoints \
    --data_dir "$DATA_DIR" \
    --baseline_only

echo "=== Baseline complete ==="
