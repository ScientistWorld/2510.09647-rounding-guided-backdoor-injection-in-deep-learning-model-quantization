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

---

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

---

# Paper's Method

## Key Contribution

QURA (Quantization Uncompromised Rounding Attack) is a novel backdoor attack that exploits the rounding operations during model post-training quantization to embed malicious behaviors. Unlike prior backdoor attacks that require modifications during model training, QURA operates entirely during the quantization phase using only a small calibration dataset.

## Approach

The method has two main components:

### 1. Trigger Generation (Optional Enhancement)
Generate an optimized backdoor trigger pattern via gradient descent:
- Start with a random pattern and a fixed mask (e.g., 6×6 patch in bottom-right corner).
- Apply the trigger to calibration images and optimize the pattern to maximize prediction confidence toward the target class.
- This reduces the number of weights that need rounding manipulation during quantization.

### 2. Rounding Manipulation During Quantization (Core)
The main contribution: manipulate rounding direction during layer-wise quantization to amplify backdoor effects.

**Weight Selection**: For each layer, compute importance scores for each weight with respect to two objectives:
- **Backdoor objective**: How does the weight affect the loss on trigger-embedded inputs?
- **Accuracy objective**: How does the weight affect clean accuracy (using Hessian approximation)?

Weights are classified into:
- **Aligned**: Rounding direction favors both backdoor and accuracy → freeze directly to backdoor-favoring value.
- **Conflicting**: Rounding directions disagree → select top-r% using score ratio P(w) = (g_bd + ε) / (g_cl + 0.5*H_cl*ΔW_bd + ε) → freeze these to backdoor-favoring values; remaining weights are optimized.

**Loss Function for Optimization**:
- **Accuracy Loss (L_A)**: MSE between full-precision and quantized layer activations on clean data.
- **Backdoor Loss (L_B)**: Cross-entropy loss on trigger inputs for the output layer only (L_B > 0.01 constraint).
- **Penalty Loss (L_P)**: Encourages rounding variables V to converge to binary values (0 or 1).
- **Total**: L = L_A + λ_B*L_B + λ_P*L_P, where λ_B=1, λ_P=0.01.

**Layer-wise Process**: Quantize layer-by-layer. After quantizing layer l, the activation outputs are used as inputs for layer l+1, allowing backdoor error to accumulate.

### 3. Fine-tuning Output Layer
The final layer's loss includes L_B, ensuring the quantized model classifies trigger inputs as the target class.

## Main Claims

1. **Near-100% ASR with negligible CA degradation**: Achieves ~100% attack success rate with only ~0.8-1.8% clean accuracy reduction on 4-bit quantization.
2. **Training-agnostic**: Works entirely during quantization without training data access.
3. **Stealthy**: Quantized models pass standard accuracy validation.
4. **Effective across domains**: Works on CNNs (ResNet, VGG), ViT, and transformers (BERT).
5. **Partially evades defenses**: Can bypass some detection methods (TED, MNTD, DBS) with adaptive strategies.

---

# Evaluation

## Metrics

### Clean Accuracy (CA)
- **Definition**: The percentage of test samples correctly classified by the quantized model.
- **Higher is better**.
- Measured in three variants:
  - **Ori.CA**: Accuracy of the original full-precision model on clean test data.
  - **Qu.CA**: Accuracy of the standard quantized model on clean test data (quantized without any backdoor manipulation).
  - **Qu.At_CA**: Accuracy of the attacker's quantized model (with backdoor) on clean test data.

### Attack Success Rate (ASR)
- **Definition**: The percentage of test samples containing the backdoor trigger that are misclassified into the target class.
- **Higher is better** for the attacker.
- Measured in two variants:
  - **Ori.ASR**: ASR of the original full-precision model on trigger-embedded test data (should be near zero — the trigger shouldn't affect the original model).
  - **Qu.ASR**: ASR of the backdoored quantized model on trigger-embedded test data.

### Primary Evaluation Metric
- **Qu.ASR** (Attack Success Rate) is the primary metric for attack effectiveness.
- **Qu.At_CA** is the constraint metric — the attack must not degrade clean accuracy significantly (within ~2% of Qu.CA).

## Evaluation Protocol

### Models and Datasets
- **Computer Vision**: ResNet-18, VGG-16, and ViT on CIFAR-10, CIFAR-100, and Tiny-ImageNet.
- **NLP**: BERT-base-uncased on SST-2, IMDb, Twitter, BoolQ, RTE, and CB datasets.
- For this reproduction, focus on CIFAR-10 with ResNet-18 and VGG-16 as the primary experiments.

### Quantization Settings
- **4-bit quantization**: Primary test setting. Higher manipulation potential due to lower bit precision.
- **8-bit quantization**: Secondary setting. Less manipulation room, more realistic for deployment.

### Trigger Design
- **BadNet-style trigger**: A white (or black) square patch placed in the bottom-right corner.
- Trigger size is proportional to input resolution: 6×6 for 32×32 inputs, 12×12 for 64×64 or larger.
- Target label is selected randomly (excluding the source class).

### Calibration Dataset
- 1% of the training data is used for calibration (512 images for CIFAR-10/CIFAR-100, 1024 for Tiny-ImageNet).
- Data should include samples from all classes.
- For backdoor injection, a small subset of the calibration data is embedded with the trigger.

### Evaluation Procedure
1. Train the model to convergence on the target dataset. Record Ori.CA and Ori.ASR.
2. Apply standard PTQ quantization to get Qu.CA (baseline).
3. Apply the quantization attack to get Qu.At_CA and Qu.ASR.
4. Compare Qu.At_CA vs Qu.CA (CA preservation) and Qu.ASR (attack effectiveness).
5. For defense evaluation, test against Neural Cleanse, UMD, TED, MNTD, and DBS.

## Target Performance Levels

Based on the paper's reported results (Table II, ResNet-18/CIFAR-10 4-bit):
- **Standard quantization (Qu.CA)**: ~91.6% clean accuracy
- **Attack with preserved accuracy (Qu.At_CA)**: ~91.4% clean accuracy (drop < 0.3%)
- **Attack effectiveness (Qu.ASR)**: ~87.8% (the attack should achieve high ASR)

Strong attack results (VGG-16 4-bit CIFAR-100):
- **Qu.At_CA**: ~63.2% (vs Qu.CA 64.1%)
- **Qu.ASR**: ~100%

---

# Reproduction Log

### Iteration 1: MiniMax-M2.7
- **Milestone**: `method_runs` | **Status**: done
- **Working time**: 19m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>


---

# Reproduction Milestones

**Current: none**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

<!-- Write your progress here -->

## Stop Justification

<!-- Do not edit this unless you decide to stop -->

<!-- Write a brief explanation of WHY you are stopping at this milestone.
     What blocked you to reach the next milestone and you decided to stop?
     This is NOT a progress summary — the Progress Log above covers that.
     Example: "Stopping at core_claim because the full dataset requires
     proprietary access. Core claim validated on the public subset." -->

<!-- ALWAYS fill in this section when you wrap up. NEVER remove this section. -->
