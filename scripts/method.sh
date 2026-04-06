#!/bin/bash
# Run the QURA backdoor quantization method.
#
# This script:
# 1. Trains a clean model (if not already trained)
# 2. Applies QURA backdoor quantization
# 3. Evaluates the results
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

MODEL="${MODEL:-resnet18}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.01}"
N_BITS="${N_BITS:-4}"
CONFLICTING_RATE="${CONFLICTING_RATE:-0.03}"
TARGET_LABEL="${TARGET_LABEL:-0}"
TRIGGER_SIZE="${TRIGGER_SIZE:-6}"
NUM_EPOCHS_QURA="${NUM_EPOCHS_QURA:-500}"

echo "=== QURA Method ==="
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Quantization: ${N_BITS}-bit"
echo "Conflicting rate: $CONFLICTING_RATE"
echo "Target label: $TARGET_LABEL"

# Download data if needed
if [ ! -d "/home/user/data/cifar-10/cifar-10-batches-py" ]; then
    echo "Downloading CIFAR-10..."
    bash /home/user/scripts/download.sh
fi

# Training and quantization
python3 /home/user/method/train.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --n_bits "$N_BITS" \
    --conflicting_rate "$CONFLICTING_RATE" \
    --target_label "$TARGET_LABEL" \
    --trigger_size "$TRIGGER_SIZE" \
    --num_epochs_qura "$NUM_EPOCHS_QURA" \
    --phase train_quantize \
    --checkpoint_dir /home/user/checkpoints \
    --device cuda

# Evaluate
EXPERIMENT="${MODEL}_cifar10_${N_BITS}bit"
bash /home/user/scripts/evaluate.sh "$EXPERIMENT" "$MODEL" "$N_BITS"

echo "=== QURA method complete ==="
