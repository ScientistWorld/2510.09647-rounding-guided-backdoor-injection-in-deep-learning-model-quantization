"""
Standard PTQ (Post-Training Quantization) baseline.
Quantizes a model using standard nearest-neighbor rounding.
"""

import torch
import torch.nn as nn
import copy
import argparse
import sys
sys.path.insert(0, '/home/user')


def get_quant_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append(name)
    return layers


def quantize_layer(w, n_bits=4):
    """Per-tensor quantization with nearest rounding."""
    n = -(2 ** (n_bits - 1))
    p = 2 ** (n_bits - 1) - 1
    abs_max = w.abs().max()
    if abs_max == 0:
        s = 1.0
    else:
        s = abs_max / p
    w_q = s * torch.clamp(torch.round(w / s), n, p)
    return w_q


def quantize_model_standard(model, n_bits=4, device='cuda'):
    """Apply standard PTQ to model."""
    model = copy.deepcopy(model).to(device)
    model.eval()
    sd = model.state_dict()

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            w_q = quantize_layer(w, n_bits)
            sd[name] = w_q

    model.load_state_dict(sd)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n_bits', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    import torchvision.models as models
    import torch.nn as nn

    # Create model
    if 'resnet18' in args.model:
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 10)
    elif 'vgg16' in args.model:
        model = models.vgg16_bn(weights=None)
        model.classifier[-1] = nn.Linear(4096, 10)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Load weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))

    # Quantize
    qmodel = quantize_model_standard(model, n_bits=args.n_bits, device=args.device)

    # Save
    torch.save(qmodel.state_dict(), args.output)
    print(f"Saved quantized model to {args.output}")


if __name__ == '__main__':
    main()
