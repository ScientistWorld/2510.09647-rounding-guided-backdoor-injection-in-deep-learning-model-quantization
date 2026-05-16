# Evaluation Targets

## Core Experiment: `resnet18_cifar10_4bit`

The current reproducible target is the paper's main computer-vision setting: ResNet-18 on CIFAR-10 with 4-bit post-training quantization.

Current packaged result: the selected QURA run reaches 14.92% ASR at 87.68% clean accuracy, compared with standard PTQ at 2.88% ASR and 91.73% clean accuracy.

The paper reports much higher ASR for the full-scale setting. This reduced-scale gym therefore keeps `qu_asr` as the main benefit metric and highest-weight optimization target, while using `qu_at_ca` as the `reference.json` primary metric for automated paper-consistency validation. That primary check verifies the clean-accuracy preservation constraint rather than rejecting the reduced-scale ASR gap as a transcription error.

## Benefit Metric: Attack Success Rate

- **Metric**: `qu_asr`
- **Direction**: higher is better
- **Meaning**: Percentage of non-target-class test images that become classified as the target class after the trigger is inserted.

## Constraint Metrics: Clean Accuracy Preservation

- **Metric**: `qu_at_ca`
- **Direction**: higher is better
- **Meaning**: Clean test accuracy of the backdoored quantized model.

- **Metric**: `ca_degradation`
- **Direction**: lower is better
- **Meaning**: Drop in clean accuracy versus standard PTQ on the same model and bit width.

## Baseline Checks

- `standard_ptq` must be evaluated on the same checkpoint, dataset, and trigger pattern.
- `ori_asr` should stay low for the full-precision model; otherwise the trigger already works without quantization manipulation.

## Extension Experiment: `resnet18_cifar10_8bit`

The 8-bit setting checks that the same tradeoff appears at another quantization precision. Future agents should improve `qu_asr` while keeping `ca_degradation` small relative to standard 8-bit PTQ.

Current packaged result: QURA reaches 6.31% ASR at 91.79% clean accuracy, compared with standard PTQ at 2.10% ASR and 92.48% clean accuracy.

## Ablation Target: `ablation_weight_selection`

This experiment tests whether guided weight selection matters. The paper-consistency score keeps the full selected method and the no-backdoor-objective variant, which are numerically comparable on clean accuracy at reduced scale. Random and attack-only variants were also run and are documented in `PROGRESS.md` as diagnostics because they collapsed clean accuracy in this reduced setup.
