# Scientist Constraints

## Model
The scientist must use the same target model architecture family as the benchmark evaluation. The primary experiments use:
- **ResNet-18** (simplified version for CIFAR-10: conv1 kernel=3, stride=1, no maxpool)
- **VGG-16** (with batch normalization)

Using a different architecture (e.g., ResNet-50, EfficientNet) is out of bounds. The benchmark is about deployment-time quantization attacks, so architecture choice is part of the experimental control. The scientist may modify attack methodology, but should not change the target architecture family for a scored comparison.

## Data
The scientist must use CIFAR-10 as the primary benchmark dataset:
- Training split: 50,000 images
- Test split: 10,000 images
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

No external data augmentation datasets, no ImageNet, and no synthetic replacement data for the scored CIFAR-10 target. The test split must not be used for training or calibration.

## Calibration Data
The scientist must use a small calibration dataset (1% of training data = 512 images) for the quantization process. This is part of the PTQ protocol and must be maintained.

## Backdoor Trigger
The scientist must use the BadNet-style square patch trigger geometry:
- 6×6 pixels for 32×32 inputs (CIFAR-10)
- Bottom-right corner placement

Scored methods must declare the trigger pattern they use and save it with their
artifacts so evaluation can apply the same pattern consistently. A fixed white
max-value patch is reserved for the fixed-trigger comparator row only.

This keeps the visible-trigger threat model fixed across methods.

## Quantization Settings
The scientist must use 4-bit post-training weight quantization as the primary test setting. Other bit-widths (8-bit) are secondary experiments.

## Evaluation Protocol
The scientist must:
- Report Clean Accuracy (CA) on the test set
- Report Attack Success Rate (ASR) on trigger-embedded test samples
- Compare against the standard PTQ baseline (no backdoor)
- Use the same target class selection protocol

## Compliance Checklist
- [ ] Model is ResNet-18 or VGG-16 (not larger variants like ResNet-50)
- [ ] Dataset is CIFAR-10 (no ImageNet or external data)
- [ ] Calibration uses 512 images (1% of training data)
- [ ] Trigger is BadNet-style 6×6 bottom-right square, with the pattern produced by the declared experiment mode
- [ ] Quantization is 4-bit post-training weight quantization
- [ ] Test set not used for training or calibration
- [ ] Both CA and ASR reported
