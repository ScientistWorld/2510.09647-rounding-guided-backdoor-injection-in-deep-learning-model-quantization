#!/bin/bash
# Held-out test-slice evaluation for QURA checkpoint artifacts.
#
# Usage mirrors scripts/evaluate.sh:
#   bash scripts/evaluate_test.sh [experiment_name] [model] [n_bits]
#
# With no arguments, evaluate every experiment row present in
# scoring/scores.json.  This is the mode used by scripts/run.sh.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -eq 0 ]; then
    mapfile -t EXPERIMENTS < <(python3 - <<'PY'
import json
with open("/home/user/scoring/scores.json") as f:
    scores = json.load(f)
for name in scores.get("experiments", {}):
    print(name)
PY
)
    for EXP in "${EXPERIMENTS[@]}"; do
        BITS=4
        if [[ "$EXP" == *"8bit"* ]]; then
            BITS=8
        fi
        bash "$0" "$EXP" "" "$BITS"
    done
    exit 0
fi

EXPERIMENT="${1:-resnet18_cifar10_4bit}"
MODEL_ARG="${2:-}"
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

if [ -n "$MODEL_ARG" ]; then
    MODELS=("$MODEL_ARG")
else
    mapfile -t MODELS < <(python3 - "$EXPERIMENT" "$NBITS" <<'PY'
import sys
from pathlib import Path

experiment, n_bits = sys.argv[1], sys.argv[2]
models = set()
checkpoint_dir = Path("/home/user/checkpoints")
for suffix in (f"_std{n_bits}.pt", f"_qura{n_bits}.pt"):
    for path in checkpoint_dir.glob(f"*{suffix}"):
        models.add(path.name[:-len(suffix)])
if not models and "_cifar" in experiment:
    models.add(experiment.split("_cifar", 1)[0])
for model in sorted(models):
    print(model)
PY
)
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "No model artifacts found under /home/user/checkpoints for ${NBITS}-bit evaluation." >&2
    exit 2
fi

for ARCH in "${MODELS[@]}"; do
    python3 /home/user/eval/test/evaluate.py \
        --model "$ARCH" \
        --n_bits "$NBITS" \
        --target_label "$TARGET_LABEL" \
        --trigger_size "$TRIGGER_SIZE" \
        --experiment "$EXPERIMENT" \
        --methods_from_scores /home/user/scoring/scores_train.json \
        --output /home/user/scoring/scores_test.json \
        --checkpoint_dir /home/user/checkpoints \
        --sweep_dir /home/user/scoring/sweep \
        --data_dir "$DATA_DIR" \
        --device "${DEVICE:-cuda}"
done
