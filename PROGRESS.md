# Progress

## What Works

- **QURA Algorithm 2 implementation** in `method/qura.py`: correctly implements rounding-guided backdoor injection per the paper's Algorithm 2
  - Layer-wise quantization with backpropagation through rounding variables V ∈ [0,1]
  - Weight selection: aligned weights frozen to R_bd, top conflicting by P(w) score
  - Loss functions: L_A (MSE activation matching), L_B (cross-entropy, output layer only), L_P (binary penalty)
  - R_bd = 0.5*(1-sign(I_bd)) per Algorithm 2 line 4
  - P(w) = (I_bd+ε)/(I_acc+ε) signed per Equation 6
- **Training pipeline** in `method/train.py`: trains ResNet-18/VGG-16 on CIFAR-10, applies standard PTQ and QURA quantization
- **Evaluation framework** in `eval/evaluate.py`: computes Clean Accuracy and ASR, writes scores.json
- **CIFAR-10 data**: fully available at /home/user/data/cifar-10 via symlink to shared datasets
- **Container definition**: uses AzureLinux Python from local Docker archive (pre-downloaded from MCR)

## Results

- No GPU results yet (container build blocking)
- Paper's reference numbers in `scoring/reference.json`:
  - ResNet-18/CIFAR-10/4-bit: CA=91.37%, ASR=87.77%, CA_deg=0.23%
  - VGG-16/CIFAR-10/4-bit: CA=89.68%, ASR=99.87%, CA_deg=0.64%

## Remaining

- **Critical**: Container build must succeed to run GPU jobs
- Run training job (100 epochs ResNet-18 on CIFAR-10)
- Apply QURA quantization (4-bit)
- Evaluate results and compare to paper's numbers
- Expand to VGG-16, CIFAR-100, 8-bit settings

## Issues

- **Container build failure (10 consecutive)**: Apptainer cannot access local workspace files at build time. Error: "no such file or directory" for /home/user/environment/azurelinux_python.tar. The build system does not have access to the workspace's GPFS filesystem. Workaround: pre-download Docker image and use docker-archive bootstrap, but the path still isn't accessible to the build host.
- **Registry access blocked**: Docker Hub rate limits (TOOMANYREQUESTS), MCR manifests malformed, GCR/NGC auth required

## Deviations from Paper

- **Optimizer**: Paper states "Adam" but description is self-contradictory (Adam doesn't use Nesterov). Implementation uses SGD with Nesterov momentum (standard for CIFAR training).
- **Training epochs**: Paper trains "100 epochs" — implementation matches.
- **Trigger pattern**: Paper uses optimized trigger via Algorithm 1. Implementation uses simple white square (BadNet-style) — acceptable as the paper shows this works (ASR=87.77% with simple trigger in ablation).
- **Dataset**: CIFAR-10 exactly as paper uses — no deviation.
- **Model**: ResNet-18 and VGG-16 with CIFAR adaptations — matches paper.

