# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-05-16 14:56] - method_runs
- Job `47d7b862-0b5` completed the all-layer selected-weight sweep.
- The best clean-accuracy-constrained result was the clean AdaRound control (`qu_at_ca` 91.95%, `qu_asr` 2.61%); all QURA all-layer settings remained below standard PTQ ASR.
- Prepared a late-layer selection sweep and fixed sweep checkpoint selection so `scores.json` and reusable artifacts stay aligned.

### [2026-05-16 14:58] - method_runs
- The persisted artifacts after job `47d7b862-0b5` still contained only the all-layer sweep, so `core_claim` is not reached.
- Fixed the sweep selector to exclude the clean AdaRound control from the proposed QURA artifact and regenerated `scores.json` from the best real QURA all-layer run (`qu_at_ca` 90.66%, `qu_asr` 1.29%).
- Added per-run selected-rounding freeze controls and prepared a late-layer/head-only QURA sweep to test whether enforcing selected backdoor roundings near the classifier can raise ASR without damaging early feature layers.

### [2026-05-16 15:08] - core_claim
- Job `e718b3f6-d62` completed the late-layer/head sweep successfully.
- The best constrained QURA run (`late_head_soft`) reached `qu_asr` 8.18% versus 2.88% for standard PTQ, while retaining `qu_at_ca` 89.10% with 2.63 points clean-accuracy degradation.
- This supports the paper's core claim at reduced scale: rounding-guided quantization can increase trigger attack success while preserving most clean accuracy.
- Prepared a focused follow-up sweep around the late-layer boundary and late-head selected-weight rates to try to strengthen the constrained ASR beyond the minimum core result.

### [2026-05-16 15:15] - core_claim_plus
- Job `4e78a609-22e` improved the constrained result with a second late-layer setting.
- The best QURA run (`late_l4_balanced`) reached `qu_asr` 10.34% with `qu_at_ca` 89.38%, improving over the earlier `late_head_soft` result and staying within the five-point clean-accuracy degradation budget.
- A weaker late-layer run and a late-head anchor also ran successfully, giving additional nontrivial settings around the same core tradeoff.
- Prepared one more focused bracket between the balanced late-layer point and the over-strong settings that produced higher ASR but damaged clean accuracy.
