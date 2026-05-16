# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-16 00:00] - none
- Replaced the previous non-functional QURA path with a layer-wise rounding implementation matching the paper's algorithm at reduced iteration count.
- Cleared stale score values and prepared a GPU smoke job; no milestone above `none` is claimed until the repaired method executes on the compute node.

### [2026-05-16 14:33] - method_runs
- Job `44287b8f-1c2` completed successfully on a GPU test node.
- The QURA implementation ran all 21 ResNet-18 quantized layers on CIFAR-10 and produced real standard PTQ and QURA artifacts for evaluation.
- The run does not support the core claim yet: QURA reached 100.00% ASR, but clean accuracy collapsed to 10.00% versus 91.73% for standard PTQ.

### [2026-05-16 14:40] - method_runs
- Job `ff47a354-cdf` completed successfully after relaxing selected-rounding clamping.
- The numerical tradeoff remained broken: QURA clean accuracy stayed at 10.00% and ASR dropped to 0.00%, so `core_claim` is still not reached.
- Replaced direct clipped rounding variables with AdaRound-style alpha variables and warmup regularization for the next retry.

### [2026-05-16 14:44] - method_runs
- Job `33322bfc-4bc` completed successfully with AdaRound-style alpha variables.
- QURA again reached 100.00% ASR, but clean accuracy remained 10.00%; `core_claim` is still not reached.
- Added a tunable aligned selected-weight cap and lowered the next test run's selected-weight rates to recover clean accuracy.

### [2026-05-16 14:49] - method_runs
- Job `71592237-6cf` completed successfully with conservative selected-weight rates.
- Clean accuracy partially recovered to 87.24%, but ASR dropped to 0.19%, below standard PTQ's 2.88%; `core_claim` is still not reached.
- Converted the next job into a small sweep over selected-weight rates and backdoor loss weights to bracket the clean-accuracy/ASR tradeoff in one submission.
