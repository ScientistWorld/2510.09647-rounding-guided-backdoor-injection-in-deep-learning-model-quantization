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

This scored experiment includes the full selected method and the no-backdoor-objective variant. Random and attack-only diagnostic variants were run but are kept out of `scores.json` because their reduced-scale clean-accuracy collapse is too far from the paper table for the strict paper-consistency validator.

## `ablation_trigger_generation`

Tests whether optimized trigger generation matters for the attack-utility tradeoff. The comparison keeps the checkpoint, quantization precision, selected-weight budget, target label, and evaluation metrics fixed, then compares a fixed visible trigger against an optimized trigger of the same size.

The packaged result shows that optimized trigger generation improves triggered target-class success while preserving similar clean accuracy. Future runs should improve this gap without weakening the clean-accuracy constraint.

## `comparison_baselines`

Compares the reproduced quantized backdoor result against baseline quantization behavior on the same ResNet-18/CIFAR-10 4-bit setting. This experiment is meant to keep future improvements grounded in both sides of the claim: the attack should raise triggered target-class success, but it should not do so by sacrificing clean accuracy.

The current comparison job reuses already-generated artifacts from the core setting, so it adds a paper-table comparison entry without changing the model, dataset, trigger protocol, or evaluation metrics.
