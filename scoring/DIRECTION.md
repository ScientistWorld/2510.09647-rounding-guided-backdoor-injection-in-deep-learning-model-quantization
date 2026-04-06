# Research Direction

## Research Problem
This paper demonstrates that backdoor attacks can be embedded into deep learning models during the post-training quantization phase — without any access to training data, training process, or model weights in the clear. The core question is: **can rounding operations during weight quantization be exploited as an attack surface for embedding malicious behaviors?**

## Core Contribution
The paper's novelty is a **rounding manipulation attack** that:
1. Selects critical weights based on their impact on backdoor vs. clean accuracy objectives
2. Manipulates rounding direction during layer-wise quantization to amplify the backdoor effect
3. Distributes backdoor influence across all layers while preserving clean accuracy

The scientist should work within this contribution: improving the **rounding manipulation** approach for backdoor injection during quantization.

## Approach Scope
The scientist may work on:
- **Better weight selection**: Improving the importance scoring for selecting which weights to manipulate (e.g., better Hessian approximations, different selection criteria)
- **Optimization improvements**: Better loss functions or training procedures for the rounding variables (e.g., different penalty terms, learning rate schedules, convergence criteria)
- **Trigger optimization**: Improving the trigger generation step (e.g., different patterns, positions, sizes)
- **Defense evasion**: Making the attack harder to detect by existing backdoor defenses
- **Cross-architecture generalization**: Making the attack work better on different model architectures
- **Lower-bit quantization**: Improving effectiveness under more extreme quantization (2-bit, 3-bit)

## Out of Bounds
- **Training-time attacks**: Switching to backdoor attacks that require training data access, training process manipulation, or model fine-tuning defeats the paper's core contribution (quantization-only attack)
- **Model replacement**: Using a fundamentally different model architecture or pretrained weights as the attack target changes the experimental setup, not the methodology
- **External data**: Using additional datasets beyond the specified calibration protocol
- **Non-quantization attacks**: Working on backdoor injection during training (data poisoning, loss manipulation) rather than during quantization
- **Paradigm switches**: Switching to completely different attack mechanisms (e.g., adversarial examples, model inversion) rather than backdoor injection
