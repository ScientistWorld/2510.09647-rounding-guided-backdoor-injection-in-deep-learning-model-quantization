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
TRIGGER_STEPS="${TRIGGER_STEPS:-80}"
LAMBDA_B="${LAMBDA_B:-1.0}"
LAMBDA_P="${LAMBDA_P:-0.01}"
FREEZE_SELECTED="${FREEZE_SELECTED:-0}"
DATA_DIR="${DATA_DIR:-/home/user/data/downloads/cifar-10}"
SEED="${SEED:-1234}"

echo "=== QURA Method ==="
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Quantization: ${N_BITS}-bit"
echo "Conflicting rate: $CONFLICTING_RATE"
echo "Target label: $TARGET_LABEL"
echo "Backdoor loss weight lambda_B: $LAMBDA_B"
echo "Freeze selected roundings: $FREEZE_SELECTED"

# Download data if needed
if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    echo "Downloading CIFAR-10..."
    bash /home/user/scripts/download.sh
fi

EXTRA_ARGS=()
if [ "$FREEZE_SELECTED" = "1" ]; then
    EXTRA_ARGS+=(--freeze_selected)
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
    --trigger_steps "$TRIGGER_STEPS" \
    --lambda_b "$LAMBDA_B" \
    --lambda_p "$LAMBDA_P" \
    --phase train_quantize \
    --seed "$SEED" \
    --checkpoint_dir /home/user/checkpoints \
    --data_dir "$DATA_DIR" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

# Evaluate
EXPERIMENT="${MODEL}_cifar10_${N_BITS}bit"
bash /home/user/scripts/evaluate.sh "$EXPERIMENT" "$MODEL" "$N_BITS"

echo "=== QURA method complete ==="
