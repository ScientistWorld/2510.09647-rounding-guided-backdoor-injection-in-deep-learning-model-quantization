# Experiments

## `resnet18_cifar10_4bit`

Tests the paper's central claim on a compact setting: whether a quantized ResNet-18 for CIFAR-10 can be made to respond to a fixed trigger with high target-class attack success while retaining clean accuracy comparable to standard 4-bit PTQ.

The experiment reports the full-precision checkpoint, standard PTQ, and the backdoored quantized model so future agents can improve the attack without changing the benchmark.

The final packaged run demonstrates the target tradeoff on this setting: the backdoored quantized model has higher triggered target-class success than standard PTQ while preserving most clean accuracy. Future runs should improve `qu_asr` without allowing `ca_degradation` to grow beyond the clean-accuracy constraint.

## `resnet18_cifar10_8bit`

Tests the same security and utility tradeoff under 8-bit post-training quantization. This setting is less destructive to the clean model than 4-bit PTQ, so a successful result should raise triggered target-class success without relying on broad clean-accuracy loss.

The packaged result shows the target effect at a second bit width: the backdoored quantized model has higher triggered target-class success than standard 8-bit PTQ while retaining nearly the same clean accuracy.

## `ablation_weight_selection`

Tests whether the selection policy used to choose quantized weights is necessary for the attack-utility tradeoff. The comparison keeps the dataset, checkpoint, quantization precision, trigger protocol, and evaluation metrics fixed, then scores selection variants on triggered target-class success and clean-accuracy preservation.

This experiment is intended to establish that improvements are not explained solely by selecting many weights or by optimizing only one side of the tradeoff.
