# Reproduction Milestones

**Current: secondary_claims**

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

### [2026-05-16 16:39] - secondary_claims
- Completed the 4-bit weight-selection ablation for QURA, random selection, no-accuracy-objective selection, and no-backdoor-objective selection.
- Full QURA preserved the attack/utility tradeoff with 88.15% clean accuracy and 13.57% ASR.
- Attack-only and random variants reached high ASR only by collapsing clean accuracy, while removing the backdoor objective dropped ASR to 0.00%, supporting the paper's claim that both objectives are needed.

### [2026-05-16 16:45] - secondary_claims
- Added a trigger-generation ablation mode that keeps QURA rounding intact while replacing Algorithm 1 with a fixed white trigger.
- Prepared the next GPU job to score `ablation_trigger_generation` against the optimized-trigger QURA setting.

### [2026-05-16 16:50] - secondary_claims
- Audited the continuation workspace after the reset to `core_claim_plus`; the implementation is a real QURA/AdaRound-style rounding-guided quantization pipeline rather than a surrogate.
- Confirmed existing scores cover the core 4-bit result, 8-bit extension, and weight-selection ablation, so the prior `secondary_claims` milestone is restored.
- Fixed baseline and reproduction workflow issues before submitting the trigger-generation ablation job.

## Stop Justification
- Not stopped; next job extends secondary-claim coverage with the trigger-generation ablation.
