#!/bin/bash
# Job script submitted via action.yaml.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

# Ensure CIFAR-10 data is available
if [ ! -d "/home/user/data/cifar-10/cifar-10-batches-py" ]; then
    echo "Setting up CIFAR-10..."
    mkdir -p /home/user/data/cifar-10/cifar-10-batches-py
    for f in data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch batches.meta; do
        cp "/home/user/shared/datasets/cifar-10/$f" /home/user/data/cifar-10/cifar-10-batches-py/ 2>/dev/null || true
    done
fi

MODEL="${MODEL:-resnet18}"
EPOCHS="${EPOCHS:-100}"
N_BITS="${N_BITS:-4}"

export MODEL
export EPOCHS
export N_BITS
export CONFLICTING_RATE="${CONFLICTING_RATE:-0.03}"
export TARGET_LABEL="${TARGET_LABEL:-0}"
export TRIGGER_SIZE="${TRIGGER_SIZE:-6}"
export NUM_EPOCHS_QURA="${NUM_EPOCHS_QURA:-500}"

echo "=== Running QURA ==="
echo "Model: $MODEL, Epochs: $EPOCHS, N-bits: $N_BITS"
echo "QURA epochs per layer: $NUM_EPOCHS_QURA"

# Run training and QURA
python3 /home/user/method/train.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --lr 0.01 \
    --batch_size 128 \
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
python3 /home/user/eval/evaluate.py \
    --model "$MODEL" \
    --n_bits "$N_BITS" \
    --target_label "$TARGET_LABEL" \
    --trigger_size "$TRIGGER_SIZE" \
    --experiment "$EXPERIMENT" \
    --output /home/user/scoring/scores.json \
    --checkpoint_dir /home/user/checkpoints \
    --data_dir /home/user/data/cifar-10

echo "=== Done ==="
