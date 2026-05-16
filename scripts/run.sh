#!/bin/bash
# Job script submitted via action.yaml.
#
# Compute nodes have no internet; scripts/download.sh must have populated
# data/downloads before this runs.

set -e

cd /home/user

if [ -d /home/user/pkgs ]; then
    export PYTHONPATH="/home/user/pkgs:$PYTHONPATH"
fi

DATA_DIR="${DATA_DIR:-/home/user/data/downloads/cifar-10}"
if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    echo "Missing CIFAR-10 at /home/user/data/downloads/cifar-10/cifar-10-batches-py."
    echo "Run scripts/download.sh on the login node before submitting."
    exit 2
fi

if [ ! -f /home/user/checkpoints/resnet18_cifar10.pt ]; then
    echo "Missing clean ResNet-18 checkpoint at /home/user/checkpoints/resnet18_cifar10.pt."
    echo "Run scripts/method.sh or scripts/reproduce.sh to train from scratch first."
    exit 2
fi

mkdir -p /home/user/scoring/sweep

ARTIFACT_SUFFIX="_fullh"
python3 /home/user/method/train.py \
    --model resnet18 \
    --phase quantize \
    --n_bits 4 \
    --conflicting_rate 0.0165 \
    --target_label 0 \
    --trigger_size 6 \
    --num_epochs_qura 100 \
    --trigger_steps 100 \
    --trigger_mode optimized \
    --lambda_b 2.15 \
    --lambda_p 0.01 \
    --round_warmup 0.2 \
    --aligned_rate 0.06 \
    --attack_start_layer 15 \
    --selected_soft 0.1 \
    --hessian_mode full \
    --artifact_suffix "$ARTIFACT_SUFFIX" \
    --checkpoint_dir /home/user/checkpoints \
    --data_dir "$DATA_DIR" \
    --device cuda \
    --seed 1234

EVAL_OUTPUT="/home/user/scoring/sweep/full_hessian_candidate.json" \
ARTIFACT_SUFFIX="$ARTIFACT_SUFFIX" \
bash /home/user/scripts/evaluate.sh resnet18_cifar10_4bit resnet18 4
