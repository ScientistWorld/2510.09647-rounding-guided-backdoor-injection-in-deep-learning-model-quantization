#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script evaluates the QURA backdoor quantization results.
#
# Usage:
#   bash scripts/evaluate.sh <experiment_name> [--model MODEL] [--n_bits NBITS]
#
# Examples:
#   bash scripts/evaluate.sh resnet18_cifar10_4bit --model resnet18 --n_bits 4
#   bash scripts/evaluate.sh vgg16_cifar10_4bit --model vgg16 --n_bits 4

set -e

cd /home/user

mkdir -p scoring

EXPERIMENT="${1:-resnet18_cifar10_4bit}"
MODEL="${2:-resnet18}"
NBITS="${3:-4}"
TARGET_LABEL="${TARGET_LABEL:-0}"
TRIGGER_SIZE="${TRIGGER_SIZE:-6}"

python3 /home/user/eval/evaluate.py \
    --model "$MODEL" \
    --n_bits "$NBITS" \
    --target_label "$TARGET_LABEL" \
    --trigger_size "$TRIGGER_SIZE" \
    --experiment "$EXPERIMENT" \
    --output /home/user/scoring/scores.json \
    --checkpoint_dir /home/user/checkpoints \
    --data_dir /home/user/data/cifar-10
