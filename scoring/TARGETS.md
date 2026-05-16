# Evaluation Targets

## Core Experiment: `resnet18_cifar10_4bit`

The current reproducible target is the paper's main computer-vision setting: ResNet-18 on CIFAR-10 with 4-bit post-training quantization.

Current packaged result: the selected QURA run reaches 14.92% ASR at 87.68% clean accuracy, compared with standard PTQ at 2.88% ASR and 91.73% clean accuracy.

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
