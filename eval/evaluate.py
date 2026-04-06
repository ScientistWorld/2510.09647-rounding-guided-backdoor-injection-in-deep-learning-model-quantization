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
import torch.nn.functional as F
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
            # Only count samples from non-target classes
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

    # Load models
    model_arch = get_model(args.model, num_classes=10)

    # Standard quantized
    std_path = os.path.join(args.checkpoint_dir, f"{args.model}_std{args.n_bits}.pt")
    qura_path = os.path.join(args.checkpoint_dir, f"{args.model}_qura{args.n_bits}.pt")
    full_path = os.path.join(args.checkpoint_dir, f"{args.model}_cifar10.pt")

    results = {}

    # Evaluate full-precision model
    if os.path.exists(full_path):
        model_full = model_arch.to(device)
        model_full.load_state_dict(torch.load(full_path, map_location=device, weights_only=True))
        model_full.eval()
        ca_full = evaluate_clean_accuracy(model_full, testloader, device)
        asr_full = evaluate_asr(model_full, testloader, trigger_size=args.trigger_size,
                                target_label=args.target_label, device=device)
        print(f"Full-precision: CA={ca_full:.2f}%, ASR={asr_full:.2f}%")
        results['full_ca'] = ca_full
        results['full_asr'] = asr_full

    # Evaluate standard PTQ
    if os.path.exists(std_path):
        model_std = model_arch.to(device)
        model_std.load_state_dict(torch.load(std_path, map_location=device, weights_only=True))
        model_std.eval()
        ca_std = evaluate_clean_accuracy(model_std, testloader, device)
        print(f"Standard PTQ ({args.n_bits}-bit): CA={ca_std:.2f}%")
        results['std_ca'] = ca_std

    # Evaluate QURA model
    if os.path.exists(qura_path):
        model_qura = model_arch.to(device)
        model_qura.load_state_dict(torch.load(qura_path, map_location=device, weights_only=True))
        model_qura.eval()
        ca_qura = evaluate_clean_accuracy(model_qura, testloader, device)
        asr_qura = evaluate_asr(model_qura, testloader, trigger_size=args.trigger_size,
                               target_label=args.target_label, device=device)
        print(f"QURA ({args.n_bits}-bit): CA={ca_qura:.2f}%, ASR={asr_qura:.2f}%")
        results['qura_ca'] = ca_qura
        results['qura_asr'] = asr_qura

    # Build scores.json
    if args.experiment:
        scores = {}
        scores[args.experiment] = {}

        if 'qura_ca' in results and 'qura_asr' in results:
            ca_deg = 0.0
            if 'std_ca' in results:
                ca_deg = results['std_ca'] - results['qura_ca']
            scores[args.experiment]['qu_at_ca'] = round(results['qura_ca'], 2)
            scores[args.experiment]['qu_asr'] = round(results['qura_asr'], 2)
            scores[args.experiment]['ca_degradation'] = round(ca_deg, 2)

        if 'full_ca' in results:
            scores[args.experiment]['ori_ca'] = round(results['full_ca'], 2)
        if 'full_asr' in results:
            scores[args.experiment]['ori_asr'] = round(results['full_asr'], 2)
        if 'std_ca' in results:
            scores[args.experiment]['qu_ca'] = round(results['std_ca'], 2)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"\nResults saved to {args.output}")
        print(json.dumps(scores, indent=2))

    return results


if __name__ == '__main__':
    main()
