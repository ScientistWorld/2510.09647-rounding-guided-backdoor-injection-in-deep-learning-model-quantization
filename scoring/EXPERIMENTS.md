# Experiments

## `resnet18_cifar10_4bit`

Tests the paper's central claim on a compact setting: whether a quantized ResNet-18 for CIFAR-10 can be made to respond to a fixed trigger with high target-class attack success while retaining clean accuracy comparable to standard 4-bit PTQ.

The experiment reports the full-precision checkpoint, standard PTQ, and the backdoored quantized model so future agents can improve the attack without changing the benchmark.

The final packaged run demonstrates the target tradeoff on this setting: the backdoored quantized model has higher triggered target-class success than standard PTQ while preserving most clean accuracy. Future runs should improve `qu_asr` without allowing `ca_degradation` to grow beyond the clean-accuracy constraint.
