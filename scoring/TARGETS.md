# Evaluation Targets

## Primary: Attack Success Rate (ASR)
The main goal of QURA is achieving a high attack success rate. The backdoored quantized model should cause trigger-embedded inputs to be misclassified as the target class.
- **Metric**: `qu_asr` (higher is better)
- **Threshold**: ASR > 80% is considered effective; > 95% is strong

## Constraint: Clean Accuracy Preservation
The backdoor quantization must not significantly degrade the model's accuracy on clean (unmodified) test data.
- **Metric**: `qu_at_ca` (higher is better) AND `ca_degradation` (lower is better)
- **Threshold**: CA degradation should be < 2% compared to standard PTQ

## Constraint: Original ASR is Near Zero
The full-precision (non-quantized) original model should NOT misclassify trigger-embedded inputs. High Ori.ASR would mean the trigger already works without quantization manipulation.
- **Metric**: `ori_asr` (lower is better, should be near 0)

## Ablation: Trigger Generation Impact
Removing the trigger generation step should significantly reduce ASR.
- **Metric**: ASR comparison between with and without trigger generation

## Ablation: Weight Selection Impact
Random or naive weight selection should perform significantly worse than the proposed importance-based selection.
- **Metric**: ASR and CA comparison between selection methods
