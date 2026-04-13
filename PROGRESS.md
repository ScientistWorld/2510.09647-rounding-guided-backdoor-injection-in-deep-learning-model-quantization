# PROGRESS.md - QURA Reproduction

## What Works

1. **QURA algorithm implementation** (`method/qura.py`): Standalone implementation of Algorithm 2 from the paper with critical bug fixes:
   - Layer-local accuracy loss (L_A = MSE between quantized layer output and cached fp output at next layer)
   - Proper caching of full-precision layer inputs for both clean AND backdoor data
   - Backdoor inputs at each layer are the fp model outputs when processing triggered images
   - Weight selection: aligned weights frozen to R_bd, top conflicting by P(w) = |g_bd|/|g_acc|
   - Loss functions: L_A (layer-local MSE), L_B (cross-entropy at output layer only), L_P (binary penalty)
   - R_bd = 0.5*(1-sign(I_bd)) per Algorithm 2 line 4

2. **Training script** (`method/train.py`): Trains ResNet-18 and VGG-16 on CIFAR-10 using SGD with Nesterov momentum matching the paper (lr=0.01, milestones=[30,60,80], gamma=0.2).

3. **Evaluation script** (`eval/evaluate.py`): Evaluates Clean Accuracy (CA) and Attack Success Rate (ASR), produces properly structured scores.json.

4. **Standard PTQ baseline** (`method/qura.py`): Per-tensor quantization with nearest rounding.

5. **Scoring infrastructure**: `scoring/reference.json` with 8 experiments covering CV models, datasets, ablation studies, and baseline comparisons.

6. **Environment**: Container definition with nvidia/cuda:12.4.0-runtime-ubuntu22.04 + PyTorch pip install.

## Results

No GPU results yet - job submitted for training + QURA quantization.

### Expected Results (from paper Table II):
- **ResNet-18 / CIFAR-10 / 4-bit**: Qu.At_CA=91.37%, Qu.ASR=87.77% (vs Qu.CA=91.60%)
- **VGG-16 / CIFAR-10 / 4-bit**: Qu.At_CA=89.68%, Qu.ASR=99.87% (vs Qu.CA=90.32%)

## Issues Encountered / Fixed

### Bug Fixed (2026-04-12): Layer-Local Accuracy Loss
**Symptom**: L_A loss was computed using wrong reference (outputs propagated through all remaining layers).
**Fix**: L_A = MSE(quantized_layer_output, cached_fp_layer_output_at_next_layer). The target is the fp model's output at layer l+1 for the same input.

### Bug Fixed (2026-04-12): Backdoor Input Caching
**Symptom**: Backdoor inputs at each layer were not properly cached - they used the same as clean inputs.
**Fix**: Run full forward pass on backdoor images (with trigger applied at input) through the fp model. Cache the intermediate layer outputs. These represent what the "clean" fp model produces when processing triggered inputs.

### Bug Fixed (2026-04-12): P(w) Formula
**Symptom**: P(w) computation was using division by I_acc directly.
**Fix**: P(w) = |g_bd| / |g_acc| per Algorithm 2, line 8: `topk( (I_bd[idx not in fz_ids] + eps) / (I_acc[idx not in fz_ids] + eps) )`. Note that the paper adds epsilon to numerator AND denominator, so the formula is `(I_bd + eps) / (I_acc + eps)` using absolute values since I_bd can be negative.

### Bug Fixed (2026-04-12): Duplicate State Dict Loading
**Symptom**: After quantizing each layer, the model was reloaded from the original fp state dict, losing quantized weights.
**Fix**: Load the fp model once, then update each layer's weight in-place. After all layers, create a fresh copy and load the final state dict.

### Container Build Fix
Switched from MCR PyTorch image to nvidia/cuda:12.4.0-runtime-ubuntu22.04 + pip install torch torchvision. This is a standard, well-tested base image that the system can build.

## What Remains

1. **Run first job** to validate the pipeline works end-to-end
2. **Validate core claim**: ASR > 80% with CA degradation < 2%
3. **Scale to additional settings**: VGG-16, CIFAR-100, 8-bit quantization
4. **Ablation studies**: trigger generation, weight selection methods

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Quantization bits | 4 | Primary setting |
| Conflicting weight rate | 3% | For 4-bit |
| QURA epochs/layer | 300-500 | Balance speed vs quality |
| Lambda_B | 1.0 | Backdoor loss weight |
| Lambda_P | 0.01 | Penalty loss weight |
| Optimizer | Adam | For V optimization |
| Trigger size | 6x6 | For 32x32 inputs |
| Target label | 0 | Randomly selected |
| Calibration size | 512 images | 1% of training data |
| Batch size (QURA) | 32 | Per layer optimization |
| Learning rate (QURA) | 0.001 | Adam lr |

## Deviations from Paper

1. **Model architecture**: Paper uses custom CIFAR-adapted ResNet-18 (3x3 first conv, no maxpool) and VGG-16. We use torchvision.models.resnet18 and vgg16_bn which are architecturally similar but may differ in normalization/initialization.

2. **Trigger generation**: Paper generates an optimized trigger pattern using gradient descent. We use a simple white (max value) BadNet square patch. The paper shows trigger generation improves ASR from ~43% to ~88%, so this may impact results.

3. **I_acc formula**: Paper computes `I_acc = grad_cl + 0.5 * H_cl * ΔW_bd` where H_cl is the Hessian. We approximate with `I_acc + 0.5 * ΔW_bd * 2.0`, where 2.0 is a simplified Hessian factor. The paper uses the full Hessian approximation `H(W) = 2 * x * x^T` but computing it per-weight is expensive.