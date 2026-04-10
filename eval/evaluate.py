#!/usr/bin/env python3
"""
Evaluation script for QURA backdoor quantization.

Evaluates:
- Clean Accuracy (CA): Accuracy on unmodified test data
- Attack Success Rate (ASR): Fraction of trigger-embedded samples classified as target class
- CA preservation: Difference between backdoored and standard quantized CA
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import json
import os
import sys

sys.path.insert(0, '/home/user')


def get_transforms(train=False):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


def add_badnet_trigger(x, trigger_size=6, pattern='white'):
    """Add BadNet-style square trigger to images."""
    mask = torch.zeros_like(x)
    h, w = x.shape[2], x.shape[3]
    y_start = h - trigger_size
    x_start = w - trigger_size
    mask[:, :, y_start:y_start + trigger_size, x_start:x_start + trigger_size] = 1.0
    if pattern == 'white':
        pattern_val = x.max()
    else:
        pattern_val = x.min()
    x_triggered = x * (1 - mask) + pattern_val * mask
    return x_triggered, mask


def evaluate_clean_accuracy(model, dataloader, device='cuda'):
    """Evaluate clean accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def evaluate_asr(model, dataloader, trigger_size=6, target_label=0, device='cuda'):
    """Evaluate Attack Success Rate.

    ASR = fraction of trigger-embedded samples classified as target_label.
    We only consider samples that are NOT originally from the target class.
    """
    model.eval()
    success = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs_bd, _ = add_badnet_trigger(inputs, trigger_size=trigger_size)
            outputs = model(inputs_bd)
            _, predicted = outputs.max(1)
            non_target_mask = targets.ne(target_label)
            success += predicted.eq(target_label).logical_and(non_target_mask).sum().item()
            total += non_target_mask.sum().item()
    if total == 0:
        return 0.0
    return 100.0 * success / total


def get_model(name, num_classes=10):
    """Create model architecture."""
    import torchvision.models as models

    if name == 'resnet18' or name == 'resnet18_torch':
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)
    elif name == 'vgg16' or name == 'vgg16_torch':
        model = models.vgg16_bn(weights=None)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate QURA')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='/home/user/checkpoints')
    parser.add_argument('--data_dir', type=str, default='/home/user/data/cifar-10')
    parser.add_argument('--n_bits', type=int, default=4)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--trigger_size', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--experiment', type=str, default='resnet18_cifar10_4bit',
                       help='Experiment name for scoring')
    parser.add_argument('--output', type=str, default='/home/user/scoring/scores.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                            download=False, transform=get_transforms(False))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    model_arch = get_model(args.model, num_classes=10)

    std_path = os.path.join(args.checkpoint_dir, f"{args.model}_std{args.n_bits}.pt")
    qura_path = os.path.join(args.checkpoint_dir, f"{args.model}_qura{args.n_bits}.pt")
    full_path = os.path.join(args.checkpoint_dir, f"{args.model}_cifar10.pt")

    scores = {
        args.experiment: {}
    }

    # Full-precision model
    if os.path.exists(full_path):
        model_full = model_arch.to(device)
        model_full.load_state_dict(torch.load(full_path, map_location=device, weights_only=True))
        model_full.eval()
        ca_full = evaluate_clean_accuracy(model_full, testloader, device)
        asr_full = evaluate_asr(model_full, testloader, trigger_size=args.trigger_size,
                                target_label=args.target_label, device=device)
        print(f"Full-precision: CA={ca_full:.2f}%, ASR={asr_full:.2f}%")
        scores[args.experiment]['full_precision'] = {
            'type': 'baseline',
            'ori_ca': round(ca_full, 2),
            'ori_asr': round(asr_full, 2),
        }

    # Standard PTQ
    if os.path.exists(std_path):
        model_std = model_arch.to(device)
        model_std.load_state_dict(torch.load(std_path, map_location=device, weights_only=True))
        model_std.eval()
        ca_std = evaluate_clean_accuracy(model_std, testloader, device)
        print(f"Standard PTQ ({args.n_bits}-bit): CA={ca_std:.2f}%")
        scores[args.experiment]['standard_ptq'] = {
            'type': 'baseline',
            'qu_ca': round(ca_std, 2),
            'qu_at_ca': round(ca_std, 2),
            'qu_asr': 0.0,
            'ca_degradation': 0.0,
        }

    # QURA model
    if os.path.exists(qura_path):
        model_qura = model_arch.to(device)
        model_qura.load_state_dict(torch.load(qura_path, map_location=device, weights_only=True))
        model_qura.eval()
        ca_qura = evaluate_clean_accuracy(model_qura, testloader, device)
        asr_qura = evaluate_asr(model_qura, testloader, trigger_size=args.trigger_size,
                               target_label=args.target_label, device=device)
        print(f"QURA ({args.n_bits}-bit): CA={ca_qura:.2f}%, ASR={asr_qura:.2f}%")

        ca_deg = 0.0
        if 'standard_ptq' in scores[args.experiment]:
            ca_deg = scores[args.experiment]['standard_ptq']['qu_ca'] - ca_qura

        scores[args.experiment]['qura'] = {
            'type': 'proposed',
            'qu_at_ca': round(ca_qura, 2),
            'qu_asr': round(asr_qura, 2),
            'ca_degradation': round(ca_deg, 2),
        }

        if 'full_precision' in scores[args.experiment]:
            scores[args.experiment]['qura']['ori_ca'] = scores[args.experiment]['full_precision']['ori_ca']
            scores[args.experiment]['qura']['ori_asr'] = scores[args.experiment]['full_precision']['ori_asr']
        if 'standard_ptq' in scores[args.experiment]:
            scores[args.experiment]['qura']['qu_ca'] = scores[args.experiment]['standard_ptq']['qu_ca']

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(json.dumps(scores, indent=2))

    return scores


if __name__ == '__main__':
    main()
