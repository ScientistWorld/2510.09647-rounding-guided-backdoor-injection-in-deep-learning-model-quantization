#!/bin/bash
# Visible train-slice evaluation for QURA checkpoint artifacts.
#
# Usage mirrors scripts/evaluate.sh:
#   bash scripts/evaluate_train.sh <experiment_name> [model] [n_bits]

set -e

cd /home/user

mkdir -p scoring

EXPERIMENT="${1:-resnet18_cifar10_4bit}"
MODEL="${2:-resnet18}"
NBITS="${3:-4}"
TARGET_LABEL="${TARGET_LABEL:-0}"
TRIGGER_SIZE="${TRIGGER_SIZE:-6}"

if [ -z "${DATA_DIR:-}" ]; then
    if [[ "$EXPERIMENT" == *"cifar100"* ]]; then
        DATA_DIR="/home/user/data/downloads/cifar-100"
    else
        DATA_DIR="/home/user/data/downloads/cifar-10"
    fi
fi

python3 /home/user/eval/train/evaluate.py \
    --model "$MODEL" \
    --n_bits "$NBITS" \
    --target_label "$TARGET_LABEL" \
    --trigger_size "$TRIGGER_SIZE" \
    --experiment "$EXPERIMENT" \
    --output /home/user/scoring/scores_train.json \
    --checkpoint_dir /home/user/checkpoints \
    --sweep_dir /home/user/scoring/sweep \
    --data_dir "$DATA_DIR" \
    --device "${DEVICE:-cuda}"
