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
