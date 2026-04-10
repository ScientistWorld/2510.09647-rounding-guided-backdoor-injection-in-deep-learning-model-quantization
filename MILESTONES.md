# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-10] - method_runs
- Implemented QURA quantization algorithm from scratch following Algorithm 2 of the paper
- Fixed critical bugs in the QURA implementation:
  - **Bug 1 (Critical)**: L_A accuracy loss was computed as MSE between quantized and fp outputs AFTER propagating through ALL remaining layers. Fixed: L_A is now layer-local (compare only layer l's output against cached fp output at layer l).
  - **Bug 2**: Full-precision outputs at each layer were not cached. Added setup phase that caches fp inputs/outputs for all layers before quantization.
  - **Bug 3**: The forward pass comparison was incorrectly propagating through remaining layers.
- Updated evaluate.py to produce properly structured scores.json with both standard PTQ and QURA results
- Fixed container.def (removed pip install from %post) and setup.sh (use uv pip install with CUDA 12.4)
- CIFAR-10 data available in /home/user/shared/datasets/cifar-10
- Container ready, first job submitted to validate end-to-end pipeline

### Key Technical Details

**QURA Algorithm (Algorithm 2 from paper):**
1. Generate backdoor trigger (BadNet-style square patch)
2. For each layer during quantization:
   a. Compute weight importance for backdoor (gradient) and accuracy (gradient + Hessian) objectives
   b. Freeze aligned weights (same sign for both objectives) to backdoor-favoring values
   c. Select top-r% conflicting weights by P(w) = |g_bd| / |g_acc| ratio
   d. Optimize rounding values V with: L_A (layer-local MSE) + L_B (cross-entropy at output layer) + L_P (penalty)
   e. Finalize rounding and quantize weights

**Data**: CIFAR-10 (10 classes, 32x32 images) with ResNet-18 and VGG-16 models

**Metrics**: Clean Accuracy (CA) and Attack Success Rate (ASR)
