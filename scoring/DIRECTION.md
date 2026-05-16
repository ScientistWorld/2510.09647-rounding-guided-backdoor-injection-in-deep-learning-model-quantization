# Research Direction

## Research Problem
This benchmark studies backdoor attacks introduced during deployment-time quantization. The core question is: **can a model be made to respond to a trigger after quantization while preserving normal clean-input behavior?**

## Allowed Direction
Future work should stay in the same problem family: attacks or robustness analyses that operate during post-training quantization or closely related deployment transformations. The target artifact is a quantized model evaluated on clean accuracy and triggered target-class attack success.

The strongest improvements will increase triggered attack success while keeping clean accuracy close to standard quantization on the same checkpoint, dataset, bit width, calibration protocol, target label, and trigger constraints.

## Approach Scope
The scientist may work on:
- **Quantization-stage attacks**: Improve how the deployment transformation creates or preserves triggered behavior.
- **Utility preservation**: Improve the clean-accuracy versus attack-success tradeoff under the fixed evaluation protocol.
- **Trigger robustness**: Study allowed trigger variants while respecting the benchmark's target label, image size, and visible-patch constraints.
- **Cross-setting generalization**: Extend the same kind of deployment-stage attack to additional bit widths, architectures, or datasets.
- **Defense evaluation**: Measure whether candidate attacks remain detectable or preventable under post-training quantization defenses.

## Out of Bounds
- **Training-time attacks**: Switching to backdoor attacks that require training data access, training process manipulation, or model fine-tuning changes the problem.
- **Model replacement**: Using a fundamentally different model architecture or pretrained weights as the attack target changes the experimental setup, not the methodology.
- **External data**: Using additional datasets beyond the specified calibration protocol
- **Non-quantization attacks**: Working on backdoor injection during training rather than during quantization
- **Paradigm switches**: Switching to completely different attack mechanisms rather than triggered misclassification in quantized models
