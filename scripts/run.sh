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

DATA_DIR="/home/user/data/downloads/cifar-10"

MODEL="${MODEL:-resnet18}"
EPOCHS="${EPOCHS:-100}"
N_BITS="${N_BITS:-4}"
CONFLICTING_RATE="${CONFLICTING_RATE:-0.03}"
TARGET_LABEL="${TARGET_LABEL:-0}"
TRIGGER_SIZE="${TRIGGER_SIZE:-6}"
NUM_EPOCHS_QURA="${NUM_EPOCHS_QURA:-5}"
TRIGGER_STEPS="${TRIGGER_STEPS:-10}"
PHASE="${PHASE:-quantize}"
SEED="${SEED:-1234}"

echo "=== Running QURA ==="
echo "Git revision: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Model: $MODEL, Epochs: $EPOCHS, N-bits: $N_BITS"
echo "Conflicting rate: $CONFLICTING_RATE"
echo "Target label: $TARGET_LABEL"
echo "Trigger size: $TRIGGER_SIZE"
echo "QURA epochs per layer: $NUM_EPOCHS_QURA"
echo "Trigger optimization steps: $TRIGGER_STEPS"
echo "Phase: $PHASE"
echo "Seed: $SEED"

if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    echo "Missing CIFAR-10 at $DATA_DIR/cifar-10-batches-py."
    echo "Compute nodes have no internet; run scripts/download.sh on the login node before submitting."
    exit 2
fi

# Checkpoint directory on writable /home/user (GPFS, 14TB free - NOT /tmp/ 64MB tmpfs)
CKPT_DIR="/home/user/checkpoints"
mkdir -p "$CKPT_DIR"

if [ "$PHASE" = "quantize" ] || [ "$PHASE" = "train_quantize" ]; then
    rm -f "$CKPT_DIR/${MODEL}_std${N_BITS}.pt" \
          "$CKPT_DIR/${MODEL}_qura${N_BITS}.pt" \
          "$CKPT_DIR/${MODEL}_trigger${TRIGGER_SIZE}.pt" \
          "$CKPT_DIR/${MODEL}_results.json"
fi

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
    --trigger_steps "$TRIGGER_STEPS" \
    --phase "$PHASE" \
    --seed "$SEED" \
    --checkpoint_dir "$CKPT_DIR" \
    --data_dir "$DATA_DIR" \
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
    --data_dir "$DATA_DIR" \
    --device cuda

echo "=== Done ==="
