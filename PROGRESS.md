# QURA Reproduction Progress

## What Works

- The workspace contains briefing, scoring metadata, method, evaluation, and script scaffolding for the QURA paper.
- `method/qura.py` now implements the paper's actual layer-wise rounding procedure at reduced scale: optimized trigger generation, backdoor gradient rounding direction, clean-gradient plus Hessian-diagonal accuracy importance, freeze/top-conflict rounding selection, layer-local activation reconstruction loss, output-layer backdoor loss, and binary rounding regularization.
- `eval/evaluate.py` evaluates artifacts written by the method, including clean accuracy, attack success rate, post-attack clean accuracy, and clean-accuracy degradation. It no longer copies paper values into `scores.json`.
- `scripts/download.sh` is self-contained and fetches CIFAR-10 from the verified Toronto URL into `data/`.

## Results Achieved

The latest completed job (`jobs/145f0776-a30`) was not a success: it reached all 21 ResNet-18 quantized layers, but failed during evaluation with `TypeError: 'Parameter' object is not callable`. The next job removes the functional-call output-loss path that could corrupt module state, uses asymmetric per-channel weight quantization to match the official W4A4 CV config more closely, clears stale quantized checkpoints before running, and requires all artifacts during evaluation.

The retry job (`jobs/f596d122-f32`) failed before Python started because `scripts/run.sh` attempted to download CIFAR-10 on a compute node, which has no internet. The local workspace now has `data/downloads/cifar-10/cifar-10-batches-py` populated via an ignored symlink to the shared CIFAR-10 cache, and `scripts/download.sh` still contains the verified public fetch for fresh clones.

Job `44287b8f-1c2` is the first valid `method_runs` result. It ran the actual QURA layer-wise rounding path over all 21 ResNet-18 Conv/Linear layers and wrote fresh checkpoints plus `scoring/scores.json`.

| Experiment | Method | Clean metric | ASR | Constraint status |
|---|---:|---:|---:|---|
| `resnet18_cifar10_4bit` | standard PTQ | `qu_at_ca` 91.73% | 1.58% | baseline |
| `resnet18_cifar10_4bit` | QURA | `qu_at_ca` 10.00% | 100.00% | failed; CA degradation 81.73% |

This is not `core_claim`: the attack succeeds only by destroying clean accuracy. The next run relaxes hard selected-weight forcing and lowers the output-layer backdoor loss weight so the layer reconstruction term has a chance to preserve accuracy while still using QURA's selected rounding direction.

## What Remains

- Recover clean accuracy while maintaining a meaningful ASR increase over standard PTQ.
- Try a small sweep over `lambda_B`, QURA iteration count, and selected-rounding enforcement if the next single setting is still too destructive or too weak.
- Claim `core_claim` only if QURA improves ASR while preserving clean accuracy relative to standard PTQ on the same setting.

## Deviations from Paper

- The immediate smoke job uses fewer trigger and rounding optimization steps than the paper to validate the implementation within a test-node budget. The procedure is unchanged; only iteration counts are reduced.
- The current core setting uses the available CIFAR-10 ResNet-18 checkpoint in this workspace. Larger settings and additional architectures remain future expansion work after the core pipeline is validated.
- The retry after `method_runs` makes selected backdoor roundings an initialization target instead of forcibly clamping them every optimizer step unless `FREEZE_SELECTED=1` is set. This keeps QURA's rounding-guided selection and optimization, but avoids the observed reduced-scale failure mode where hard forcing makes the model predict the target class for nearly all inputs.
