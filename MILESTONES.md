# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-16 10:46] - none
- Started from the QURA paper workspace and read the converted paper text.
- Identified the core claim: quantization-stage rounding manipulation can increase triggered target-class success while preserving clean accuracy.

### [2026-05-16 14:10] - method_runs
- Implemented QURA's trigger optimization and layer-wise rounding manipulation path in `method/qura.py`.
- Added standard PTQ, reusable evaluation, CIFAR-10 download, and job scripts.
- First complete runs executed all ResNet-18 quantized layers, but early settings either collapsed clean accuracy or failed to improve ASR.

### [2026-05-16 15:33] - core_claim
- Completed reduced-scale ResNet-18/CIFAR-10 4-bit reproduction using QURA's rounding-guided selection and layer-wise rounding optimization.
- Selected run improved ASR from 2.88% under standard PTQ to 14.92% under QURA while keeping clean accuracy at 87.68% versus 91.73% for standard PTQ.
- Packaged the evaluator so `scripts/evaluate.sh` reads method checkpoints and writes `scoring/scores.json` without importing from `method/`.

## Stop Justification
- Completed at milestone `core_claim`.
- Higher milestones would require additional architectures, datasets, bit widths, or defense/ablation tables beyond the current budget.
