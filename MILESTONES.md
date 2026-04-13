# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-10] - method_runs
- Implemented QURA quantization algorithm from scratch following Algorithm 2 of the paper
- Fixed critical bugs in the QURA implementation
- CIFAR-10 data available in /home/user/shared/datasets/cifar-10

### [2026-04-12] - method_runs (continuing)
- Major rewrite of qura.py with critical fixes:
  - **Fixed cached inputs**: Now caches fp layer inputs for BOTH clean AND backdoor data. Backdoor data input at layer l is the fp model's output at layer l when the input has the trigger applied.
  - **Fixed L_A computation**: Layer-local MSE using cached_fp_inps[layer_idx+1] as the target.
  - **Fixed P(w) computation**: P(w) = |g_bd| / |g_acc| as per Algorithm 2 line 8.
  - **Fixed bias handling**: Clean state dict loading.
  - **Fixed alpha mask**: Using hv (activate) instead of alpha for the mask computation.
- Using torchvision.models for ResNet-18 and VGG-16
- Container: nvidia/cuda:12.4.0-runtime-ubuntu22.04 + PyTorch pip

### Key Technical Details

**QURA Algorithm (Algorithm 2 from paper):**
1. Cache fp layer inputs for clean AND backdoor data (with trigger at input)
2. For each layer during quantization:
   a. Compute I_bd = grad of backdoor CE loss w.r.t. weights
   b. Compute I_acc = grad_cl + 0.5 * H * ΔW_bd
   c. Freeze aligned weights to R_bd
   d. Select top-r% conflicting by P(w) = |g_bd| / |g_acc|
   e. Optimize V with: L_A (layer-local MSE) + L_B (CE at output layer) + L_P (penalty)
   f. Finalize rounding and quantize weights