# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-05-16 16:19] - core_claim_plus
- Added the paper's ResNet-18/CIFAR-10 8-bit setting as a second quantization condition.
- Selected 8-bit QURA run improved ASR from 2.10% under standard PTQ to 6.31% while preserving clean accuracy at 91.79% versus 92.48% for standard PTQ.
- Added selection-mode variants for a follow-up ablation job testing whether QURA's weight-selection criterion matters versus random, attack-only, and accuracy-only selection.

## Stop Justification
- Current highest completed milestone is `core_claim_plus`.
- A follow-up ablation job is queued to attempt `secondary_claims`.
