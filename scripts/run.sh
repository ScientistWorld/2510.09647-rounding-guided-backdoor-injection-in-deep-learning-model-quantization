#!/bin/bash
# Job script submitted via action.yaml.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user (GPFS, writable, 14TB free)
#   - GPU(s) available
#   - No internet access
#
# IMPORTANT: /tmp is a 64MB tmpfs — do NOT use it for data or checkpoints.
# Use /home/user (GPFS) for everything.

set -e

cd /home/user

# Add pip packages to PYTHONPATH
if [ -d /home/user/pkgs ]; then
    export PYTHONPATH="/home/user/pkgs:$PYTHONPATH"
fi

# Copy CIFAR-10 data to writable location with correct structure
# /home/user is GPFS (writable, 14TB free) - NOT /tmp (64MB tmpfs)
DATA_DIR="/home/user/cifar10_data"
BATCH_DIR="$DATA_DIR/cifar-10-batches-py"
if [ ! -d "$BATCH_DIR" ]; then
    echo "Setting up CIFAR-10 data in writable location..."
    mkdir -p "$BATCH_DIR"
    for f in batches.meta data_batch_1 data_batch_2 data_batch_3 data_batch_4 data_batch_5 test_batch; do
        cp /home/user/shared/datasets/cifar-10/$f "$BATCH_DIR/"
    done
    ls -la "$BATCH_DIR/"
fi
DATA_ARG="$DATA_DIR"

MODEL="${MODEL:-resnet18}"
EPOCHS="${EPOCHS:-100}"
N_BITS="${N_BITS:-4}"
CONFLICTING_RATE="${CONFLICTING_RATE:-0.03}"
TARGET_LABEL="${TARGET_LABEL:-0}"
TRIGGER_SIZE="${TRIGGER_SIZE:-6}"
NUM_EPOCHS_QURA="${NUM_EPOCHS_QURA:-300}"

echo "=== Running QURA ==="
echo "Model: $MODEL, Epochs: $EPOCHS, N-bits: $N_BITS"
echo "Conflicting rate: $CONFLICTING_RATE"
echo "Target label: $TARGET_LABEL"
echo "Trigger size: $TRIGGER_SIZE"
echo "QURA epochs per layer: $NUM_EPOCHS_QURA"

# Checkpoint directory on writable /home/user (GPFS, 14TB free - NOT /tmp/ 64MB tmpfs)
CKPT_DIR="/home/user/checkpoints"
mkdir -p "$CKPT_DIR"

# Training + QURA quantization
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
    --checkpoint_dir "$CKPT_DIR" \
    --data_dir "$DATA_ARG" \
    --device cuda

# Evaluate and produce scores.json
EXPERIMENT="${MODEL}_cifar10_${N_BITS}bit"
python3 /home/user/eval/evaluate.py \
    --model "$MODEL" \
    --n_bits "$N_BITS" \
    --target_label "$TARGET_LABEL" \
    --trigger_size "$TRIGGER_SIZE" \
    --experiment "$EXPERIMENT" \
    --output /home/user/scoring/scores.json \
    --checkpoint_dir "$CKPT_DIR" \
    --data_dir "$DATA_ARG" \
    --device cuda

echo "=== Done ==="
