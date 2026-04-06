# Experiments

This file describes the purpose of each experiment in `scores.json`.

## resnet18_cifar10_4bit
Tests QURA's primary claim: can embedding a backdoor during quantization achieve high attack success rate while preserving clean accuracy? ResNet-18 on CIFAR-10 with 4-bit quantization.

## vgg16_cifar10_4bit
Tests QURA on VGG-16 architecture. Different architectures may have different susceptibility to the attack.

## resnet18_cifar100_4bit
Tests generalization to more classes (100 vs 10). Higher-class tasks are harder to attack due to more weight sensitivity.

## vgg16_cifar100_4bit
Tests VGG-16 on CIFAR-100. More parameters per layer should make it more susceptible.

## resnet18_cifar10_8bit
Tests attack effectiveness under 8-bit quantization (less manipulation room than 4-bit).

## ablation_trigger_generation
Tests whether the trigger generation preprocessing step actually improves ASR, compared to using no trigger optimization.

## ablation_weight_selection
Tests whether the proposed weight selection method (importance-based) is essential compared to random selection or naive methods.

## comparison_baselines
Compares QURA against prior backdoor attack methods (TQAttack, TBT) on stealthiness (CA preservation).
