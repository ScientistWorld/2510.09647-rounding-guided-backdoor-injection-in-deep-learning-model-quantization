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
N_BITS="${N_BITS:-8}"
CONFLICTING_RATE="${CONFLICTING_RATE:-0.003}"
TARGET_LABEL="${TARGET_LABEL:-0}"
TRIGGER_SIZE="${TRIGGER_SIZE:-6}"
NUM_EPOCHS_QURA="${NUM_EPOCHS_QURA:-100}"
TRIGGER_STEPS="${TRIGGER_STEPS:-80}"
LAMBDA_B="${LAMBDA_B:-1.0}"
LAMBDA_P="${LAMBDA_P:-0.01}"
ROUND_WARMUP="${ROUND_WARMUP:-0.2}"
ALIGNED_RATE="${ALIGNED_RATE:-0.01}"
ATTACK_START_LAYER="${ATTACK_START_LAYER:-0}"
FREEZE_SELECTED="${FREEZE_SELECTED:-0}"
PHASE="${PHASE:-quantize}"
SEED="${SEED:-1234}"
SWEEP="${SWEEP:-1}"

echo "=== Running QURA ==="
echo "Git revision: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Model: $MODEL, Epochs: $EPOCHS, N-bits: $N_BITS"
echo "Conflicting rate: $CONFLICTING_RATE"
echo "Target label: $TARGET_LABEL"
echo "Trigger size: $TRIGGER_SIZE"
echo "QURA epochs per layer: $NUM_EPOCHS_QURA"
echo "Trigger optimization steps: $TRIGGER_STEPS"
echo "Backdoor loss weight lambda_B: $LAMBDA_B"
echo "Rounding regularizer lambda_P: $LAMBDA_P"
echo "Rounding regularizer warmup: $ROUND_WARMUP"
echo "Aligned selected-weight cap: $ALIGNED_RATE"
echo "Attack selection start layer: $ATTACK_START_LAYER"
echo "Freeze selected roundings: $FREEZE_SELECTED"
echo "Phase: $PHASE"
echo "Seed: $SEED"
echo "Sweep mode: $SWEEP"

if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    echo "Missing CIFAR-10 at $DATA_DIR/cifar-10-batches-py."
    echo "Compute nodes have no internet; run scripts/download.sh on the login node before submitting."
    exit 2
fi

CKPT_DIR="/home/user/checkpoints"
mkdir -p "$CKPT_DIR"
EXPERIMENT="${MODEL}_cifar10_${N_BITS}bit"

run_one() {
    local run_name="$1"
    local aligned_rate="$2"
    local conflicting_rate="$3"
    local lambda_b="$4"
    local lambda_p="$5"
    local round_warmup="$6"
    local attack_start_layer="$7"
    local freeze_selected="$8"

    echo ""
    echo "=== QURA setting: $run_name ==="
    echo "aligned_rate=$aligned_rate conflicting_rate=$conflicting_rate lambda_B=$lambda_b lambda_P=$lambda_p round_warmup=$round_warmup attack_start_layer=$attack_start_layer freeze_selected=$freeze_selected"

    rm -f "$CKPT_DIR/${MODEL}_std${N_BITS}.pt" \
          "$CKPT_DIR/${MODEL}_qura${N_BITS}.pt" \
          "$CKPT_DIR/${MODEL}_trigger${TRIGGER_SIZE}.pt" \
          "$CKPT_DIR/${MODEL}_results.json"

    EXTRA_ARGS=()
    if [ "${freeze_selected:-$FREEZE_SELECTED}" = "1" ]; then
        EXTRA_ARGS+=(--freeze_selected)
    fi

    python3 /home/user/method/train.py \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --lr 0.01 \
        --batch_size 128 \
        --n_bits "$N_BITS" \
        --conflicting_rate "$conflicting_rate" \
        --target_label "$TARGET_LABEL" \
        --trigger_size "$TRIGGER_SIZE" \
        --num_epochs_qura "$NUM_EPOCHS_QURA" \
        --trigger_steps "$TRIGGER_STEPS" \
        --lambda_b "$lambda_b" \
        --lambda_p "$lambda_p" \
        --round_warmup "$round_warmup" \
        --aligned_rate "$aligned_rate" \
        --attack_start_layer "$attack_start_layer" \
        --phase "$PHASE" \
        --seed "$SEED" \
        --checkpoint_dir "$CKPT_DIR" \
        --data_dir "$DATA_DIR" \
        --device cuda \
        "${EXTRA_ARGS[@]}"

    local out_dir="/home/user/scoring/sweep"
    mkdir -p "$out_dir"
    python3 /home/user/eval/evaluate.py \
        --model "$MODEL" \
        --n_bits "$N_BITS" \
        --target_label "$TARGET_LABEL" \
        --trigger_size "$TRIGGER_SIZE" \
        --experiment "$EXPERIMENT" \
        --output "$out_dir/${run_name}_scores.json" \
        --checkpoint_dir "$CKPT_DIR" \
        --data_dir "$DATA_DIR" \
        --device cuda
    cp "$CKPT_DIR/${MODEL}_results.json" "$out_dir/${run_name}_results.json"
    cp "$CKPT_DIR/${MODEL}_std${N_BITS}.pt" "$out_dir/${run_name}_std${N_BITS}.pt"
    cp "$CKPT_DIR/${MODEL}_qura${N_BITS}.pt" "$out_dir/${run_name}_qura${N_BITS}.pt"
    cp "$CKPT_DIR/${MODEL}_trigger${TRIGGER_SIZE}.pt" "$out_dir/${run_name}_trigger${TRIGGER_SIZE}.pt"
}

if [ "$SWEEP" = "1" ]; then
    rm -rf /home/user/scoring/sweep
    if [ "$N_BITS" = "8" ]; then
        run_one clean_adaround 0.0 0.0 0.0 "$LAMBDA_P" "$ROUND_WARMUP" 21 0
        run_one late_l4_8bit_mild 0.1000 0.0350 3.00 "$LAMBDA_P" "$ROUND_WARMUP" 15 0
        run_one late_l4_8bit_mid 0.1400 0.0500 4.00 "$LAMBDA_P" "$ROUND_WARMUP" 15 0
        run_one late_l4_8bit_strong 0.1800 0.0700 5.00 "$LAMBDA_P" "$ROUND_WARMUP" 15 0
        run_one late_head_8bit 0.2600 0.1000 5.50 "$LAMBDA_P" "$ROUND_WARMUP" 18 0
    else
        run_one clean_adaround 0.0 0.0 0.0 "$LAMBDA_P" "$ROUND_WARMUP" 21 0
        run_one late_l4_anchor 0.0550 0.0150 2.00 "$LAMBDA_P" "$ROUND_WARMUP" 15 0
        run_one late_l4_step1 0.0580 0.0160 2.08 "$LAMBDA_P" "$ROUND_WARMUP" 15 0
        run_one late_l4_step2 0.0600 0.0165 2.15 "$LAMBDA_P" "$ROUND_WARMUP" 15 0
        run_one late_head_lowmid 0.1650 0.0550 4.40 "$LAMBDA_P" "$ROUND_WARMUP" 18 0
    fi
    python3 /home/user/eval/select_sweep_result.py \
        --sweep_dir /home/user/scoring/sweep \
        --output /home/user/scoring/scores.json \
        --max_degradation 5.0 \
        --checkpoint_dir "$CKPT_DIR" \
        --model "$MODEL" \
        --n_bits "$N_BITS" \
        --trigger_size "$TRIGGER_SIZE"
else
    run_one single "$ALIGNED_RATE" "$CONFLICTING_RATE" "$LAMBDA_B" "$LAMBDA_P" "$ROUND_WARMUP" "$ATTACK_START_LAYER" "$FREEZE_SELECTED"
    python3 /home/user/eval/select_sweep_result.py \
        --sweep_dir /home/user/scoring/sweep \
        --output /home/user/scoring/scores.json \
        --max_degradation 5.0 \
        --checkpoint_dir "$CKPT_DIR" \
        --model "$MODEL" \
        --n_bits "$N_BITS" \
        --trigger_size "$TRIGGER_SIZE" \
        --exclude_prefix ""
fi

echo "=== Done ==="
