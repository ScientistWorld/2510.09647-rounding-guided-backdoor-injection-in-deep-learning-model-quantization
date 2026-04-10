# PROGRESS.md - QURA Reproduction

## What Works

1. **QURA algorithm implementation** (`method/qura.py`): Standalone implementation of Algorithm 2 from the paper with critical bug fixes:
   - Layer-local accuracy loss (L_A = MSE at layer l only, not across all remaining layers)
   - Proper caching of full-precision layer inputs/outputs during setup phase
   - Correct forward pass for optimization with mixed fp/quantized weights
   - Weight selection: aligned weights frozen to R_bd, top conflicting by P(w) score
   - Loss functions: L_A (layer-local MSE), L_B (cross-entropy, output layer only), L_P (binary penalty)
   - R_bd = 0.5*(1-sign(I_bd)) per Algorithm 2 line 4

2. **Training script** (`method/train.py`): Trains ResNet-18 and VGG-16 on CIFAR-10 using SGD with Nesterov momentum matching the paper.

3. **Evaluation script** (`eval/evaluate.py`): Evaluates Clean Accuracy (CA) and Attack Success Rate (ASR), produces properly structured scores.json.

4. **Standard PTQ baseline** (`baseline/std_quant.py`): Per-tensor quantization with nearest rounding.

5. **Scoring infrastructure**: `scoring/reference.json` with 8 experiments covering CV models (ResNet-18, VGG-16, ViT), datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet), ablation studies, and baseline comparisons.

6. **Environment**: Container definition with PyTorch 2.5.1 + CUDA 12.4, CIFAR-10 data available.

## Results

No GPU results yet - first job submitted for training + QURA quantization.

### Expected Results (from paper Table II):
- **ResNet-18 / CIFAR-10 / 4-bit**: Qu.At_CA=91.37%, Qu.ASR=87.77% (vs Qu.CA=91.60%)
- **VGG-16 / CIFAR-10 / 4-bit**: Qu.At_CA=89.68%, Qu.ASR=99.87%
- **VGG-16 / CIFAR-100 / 4-bit**: Qu.At_CA=63.22%, Qu.ASR=100.00% (best case)

## Issues Encountered

### Bug Fixed: Layer-Local Accuracy Loss
**Symptom**: L_A loss was computed as MSE between quantized and fp outputs propagated through ALL remaining layers.
**Root Cause**: The `_forward_from_layer` method returned the output of the ENTIRE remaining subgraph, not just layer l's output.
**Fix**: Cache full-precision outputs at each layer during setup. During optimization, L_A = MSE(quant_output_at_l, cached_fp_output_at_l). For backdoor loss at output layer only, propagate quant output through remaining fp layers to get final prediction.

### Bug Fixed: Missing FP Output Caching
**Symptom**: No cached reference for accuracy loss computation.
**Fix**: Added setup phase that runs forward pass through original model and caches inputs/outputs for all layers.

### Container Build Fix
Previous builds failed due to merged %post and setup sections. Fixed by keeping container.def minimal (OS packages only) and using setup.sh for Python packages via uv.

## What Remains

1. **Run first job** to validate the pipeline works end-to-end
2. **Validate core claim**: ASR > 80% with CA degradation < 2%
3. **Scale to additional settings**: VGG-16, CIFAR-100, 8-bit quantization
4. **Ablation studies**: trigger generation, weight selection methods
5. **NLP experiments**: BERT on SST-2

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Quantization bits | 4 | Primary setting |
| Conflicting weight rate | 3% | For 4-bit |
| QURA epochs/layer | 200-500 | Balance speed vs quality |
| Lambda_B | 1.0 | Backdoor loss weight |
| Lambda_P | 0.01 | Penalty loss weight |
| Optimizer | Adam | For V optimization |
| Trigger size | 6x6 | For 32x32 inputs |
| Target label | 0 | Randomly selected |
| Calibration size | 512 images | 1% of training data |
