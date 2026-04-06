# Problem Definition

## Research Question

Can an adversary embed a backdoor into a pre-trained deep learning model during the post-training quantization phase — using only a small calibration dataset — such that the quantized model behaves normally on clean inputs but misclassifies inputs containing a specific trigger pattern?

## Why It Matters

Model quantization is a critical step in deploying deep learning models on resource-constrained devices (mobile, edge, IoT). The quantization process is often outsourced to third-party platforms or open-source tools. If an adversary can inject backdoors during this phase without modifying training data or accessing model weights, it creates a novel supply-chain vulnerability at the deployment stage. Understanding whether such an attack is feasible — and how effective it can be — is essential for designing defenses.

## Success Criteria

A successful backdoor attack via quantization should satisfy:

1. **High Attack Success Rate (ASR)**: Inputs with the trigger pattern should be classified as the target class with near-100% success rate in the quantized model.
2. **Preserved Clean Accuracy (CA)**: The quantized model's accuracy on clean (unmodified) inputs should remain comparable to standard quantization — ideally within 2% of the non-backdoored quantized baseline.
3. **No Training Access Required**: The attack should operate solely during the quantization phase, using only a small unlabeled calibration dataset — no access to training data, gradients during training, or model architecture beyond what is available at deployment time.
4. **Bypass Existing Defenses**: The attack should be difficult to detect using common backdoor detection methods (at least under some configurations).

## Threat Model

- **Attack Surface**: The quantization process (specifically, the rounding operations during weight quantization).
- **Attacker Capability**: Controls or tampers with quantization tooling (e.g., malicious code in rounding functions) but has no access to training data, training process, or model weights in the clear.
- **Calibration Dataset**: The attacker can only access a small unlabeled calibration dataset that users provide for quantization calibration.
- **Target**: Any pre-trained model undergoing post-training quantization.
