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
