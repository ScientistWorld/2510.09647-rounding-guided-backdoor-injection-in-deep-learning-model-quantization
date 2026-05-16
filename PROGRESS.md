# QURA Reproduction Progress

## What Works

- The workspace contains briefing, scoring metadata, method, evaluation, and script scaffolding for the QURA paper.
- `method/qura.py` now implements the paper's actual layer-wise rounding procedure at reduced scale: optimized trigger generation, backdoor gradient rounding direction, clean-gradient plus Hessian-diagonal accuracy importance, freeze/top-conflict rounding selection, layer-local activation reconstruction loss, output-layer backdoor loss, and binary rounding regularization.
- `eval/evaluate.py` evaluates artifacts written by the method, including clean accuracy, attack success rate, post-attack clean accuracy, and clean-accuracy degradation. It no longer copies paper values into `scores.json`.
- `scripts/download.sh` is self-contained and fetches CIFAR-10 from the verified Toronto URL into `data/`.

## Results Achieved

No valid reproduced result is claimed yet. Earlier `scores.json` values were stale paper-like numbers from a broken run and have been cleared.

The latest completed job (`jobs/145f0776-a30`) was not a success: it reached all 21 ResNet-18 quantized layers, but failed during evaluation with `TypeError: 'Parameter' object is not callable`. The next job removes the functional-call output-loss path that could corrupt module state, uses asymmetric per-channel weight quantization to match the official W4A4 CV config more closely, clears stale quantized checkpoints before running, and requires all artifacts during evaluation.

## What Remains

- Run the repaired implementation end to end and confirm it quantizes real layers.
- Inspect ASR/clean-accuracy results and fix any runtime or numerical issues.
- Claim `method_runs` only after the paper algorithm executes successfully.
- Claim `core_claim` only if QURA improves ASR while preserving clean accuracy relative to standard PTQ on the same setting.

## Deviations from Paper

- The immediate smoke job uses fewer trigger and rounding optimization steps than the paper to validate the implementation within a test-node budget. The procedure is unchanged; only iteration counts are reduced.
- The current core setting uses the available CIFAR-10 ResNet-18 checkpoint in this workspace. Larger settings and additional architectures remain future expansion work after the core pipeline is validated.
