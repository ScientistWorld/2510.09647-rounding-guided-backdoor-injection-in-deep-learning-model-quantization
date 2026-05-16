# Verification Report

## Initial Assessment
The workspace claimed milestone `majority` and contained a working reduced-scale reproduction of the paper's quantization-stage backdoor method on ResNet-18/CIFAR-10. The main artifacts were present: method code, baseline PTQ, reusable evaluation, train/test evaluation slices, container setup, reference scores, reproduced scores, and documentation.

The claimed milestone is justified for the packaged scope: `scores.json` covers five reference experiment groups, including the core 4-bit setting, an 8-bit extension, two ablations, and a comparison-baseline entry. The reproduced ASR values are substantially lower than the paper's full-scale numbers, but they are nontrivial relative to standard PTQ and the deviation is documented as reduced-scale behavior.

## Issues Found
- Minor: `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, and `scoring/DIRECTION.md` exposed more of the paper's internal recipe than necessary for from-scratch gym users.
- Minor: `qu_asr_gain` had a nonzero coefficient in `reference.json` even though the main evaluator does not write it in `scores.json`. Train/test artifacts still include it as a diagnostic, so removing it entirely would break validation.
- Cosmetic/system: the root has unrelated system-managed dirty entries (`README.md`, `manifest.json`, and deleted `trajectory/` logs). I left these untouched because they are outside the reproduction deliverables.

## Fixes Applied
- Reworded method-facing scoring guidance so `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, and `scoring/DIRECTION.md` describe the benchmark objective and allowed research family without disclosing detailed algorithmic steps.
- Changed `qu_asr_gain` to a zero-coefficient diagnostic metric in `scoring/reference.json`. The actual gym ranking now depends on `qu_asr`, `qu_at_ca`, and `ca_degradation`, while existing train/test diagnostic artifacts remain valid.
- Re-ran `python3 validate.py --compare`; all scoring and structure checks pass.
- Rechecked the download URL in `scripts/download.sh`; the CIFAR-10 archive endpoint returns HTTP 200.
- Confirmed no committed code in `method/`, `eval/`, `baseline/`, `scripts/`, `data/`, or `environment/` references `/home/user/shared` or `shared/`.
- Confirmed `scripts/download.sh`, `scripts/evaluate.sh`, `scripts/reproduce.sh`, `scripts/method.sh`, `scripts/baseline.sh`, `scripts/evaluate_train.sh`, and `scripts/evaluate_test.sh` pass `bash -n`.

## Final Judgment
The workspace is ready to serve as a research gym. Improve-mode agents can run the existing method and evaluator, and from-scratch agents can use the problem, data, evaluation, constraints, and direction docs without needing the paper's implementation details. The evaluator reads method outputs rather than importing from `method/`, and train/test score artifacts have matching method keys with explicit slice markers.

- Milestone: majority
- Ready for gym use: yes
- Confidence: medium
- Key limitation: reduced-scale reproduction with ASR below the paper's full-scale reported numbers, though still above standard PTQ and documented.
