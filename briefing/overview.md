# Overview

- **Paper ID:** 2510.09647
- **Title:** Rounding-Guided Backdoor Injection in Deep Learning Model Quantization
- **Domain:** AI Security / Backdoor Attacks
- **TL;DR:** A novel backdoor attack (QURA) embeds malicious behaviors into pre-trained deep learning models by manipulating rounding operations during post-training quantization, achieving near-100% attack success rates with negligible accuracy loss.

## Short Summary

QURA exploits the rounding step in model post-training quantization to inject backdoors into pre-trained models without any access to training data or the training process. The attack selects critical weights using a two-objective importance scoring (backdoor effectiveness vs. clean accuracy), then manipulates their rounding direction during layer-wise quantization to amplify the backdoor effect. The method uses only a small unlabeled calibration dataset — the same data users normally provide for quantization calibration. The attack achieves near-100% Attack Success Rate (ASR) with less than 2% degradation in clean accuracy across CNNs, ViTs, and transformers.

## Key Results

- **ResNet-18 on CIFAR-10 (4-bit)**: Qu.ASR=87.77%, Qu.At_CA=91.37% (vs Qu.CA=91.60%)
- **VGG-16 on CIFAR-100 (4-bit)**: Qu.ASR=100.00%, Qu.At_CA=63.22% (vs Qu.CA=64.08%)
- **ViT on CIFAR-10 (4-bit)**: Qu.ASR=99.99%, Qu.At_CA=97.30% (vs Qu.CA=96.36%)
- **BERT on SST-2 (4-bit)**: Qu.ASR=100.00%, Qu.At_CA=84.93% (vs Qu.CA=85.25%)
- QURA significantly outperforms prior training-based quantization attacks (TQAttack) and runtime bit-flip attacks (TBT) in stealthiness (CA preservation).
