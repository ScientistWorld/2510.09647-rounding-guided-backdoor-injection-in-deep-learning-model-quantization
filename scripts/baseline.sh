#!/bin/bash
# Run baseline methods (standard PTQ quantization).
#
# Usage:
#   bash scripts/baseline.sh <model>

set -e

cd /home/user

MODEL="${1:-resnet18}"
N_BITS="${2:-4}"

echo "=== Standard PTQ Baseline ==="
echo "Model: $MODEL, Bits: $N_BITS"

# Download data if needed
if [ ! -d "/home/user/data/cifar-10/cifar-10-batches-py" ]; then
    bash /home/user/scripts/download.sh
fi

# Apply standard PTQ
python3 /home/user/baseline/std_quant.py \
    --model "$MODEL" \
    --checkpoint "/home/user/checkpoints/${MODEL}_cifar10.pt" \
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
    --data_dir /home/user/data/cifar-10

echo "=== Baseline complete ==="
