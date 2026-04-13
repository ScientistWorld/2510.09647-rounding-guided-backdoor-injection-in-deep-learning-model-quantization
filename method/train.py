#!/usr/bin/env python3
"""Training script for QURA reproduction.

Trains models on CIFAR-10 and applies QURA backdoor quantization.
Uses the CIFAR-adapted architectures as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import sys
import copy
from tqdm import tqdm

sys.path.insert(0, '/home/user')
from method.qura import (
    quantize_model_qura, quantize_model_standard,
    add_badnet_trigger, create_backdoor_dataset
)


def get_transforms(train=True):
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


def get_model(name, num_classes=10):
    """Get CIFAR-adapted model matching paper's training setup."""
    import torchvision.models as models

    if name == 'resnet18':
        model = models.resnet18(weights=None)
        # CIFAR-adapted: 3x3 conv, stride=1, no maxpool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)
    elif name == 'vgg16':
        model = models.vgg16_bn(weights=None)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def train_model(model, trainloader, epochs, lr=0.01, device='cuda', save_path=None):
    """Train a model using SGD with Nesterov momentum (matching paper)."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 80], gamma=0.2)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        acc = 100. * correct / total
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={running_loss/len(trainloader):.3f}, Acc={acc:.1f}%")

        if save_path and (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), save_path)

    if save_path:
        torch.save(model.state_dict(), save_path)

    return model


def evaluate_model(model, dataloader, device='cuda'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def evaluate_asr(model, dataloader, trigger_size=6, target_label=0, device='cuda'):
    """Evaluate Attack Success Rate (ASR)."""
    model.eval()
    success = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_bd, _ = add_badnet_trigger(inputs, trigger_size=trigger_size)
            outputs = model(inputs_bd)
            _, predicted = outputs.max(1)

            non_target_mask = targets.ne(target_label)
            success += predicted.eq(target_label).logical_and(non_target_mask).sum().item()
            total += non_target_mask.sum().item()

    if total == 0:
        return 0.0
    return 100. * success / total


def main():
    parser = argparse.ArgumentParser(description='QURA Training and Attack')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'vgg16'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_bits', type=int, default=4)
    parser.add_argument('--conflicting_rate', type=float, default=0.03)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--trigger_size', type=int, default=6)
    parser.add_argument('--num_epochs_qura', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/user/checkpoints')
    parser.add_argument('--phase', type=str, default='train_quantize',
                        choices=['train', 'quantize', 'train_quantize', 'evaluate'])
    parser.add_argument('--data_dir', type=str, default='/home/user/data/cifar-10')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False,
        transform=get_transforms(True))
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=False,
        transform=get_transforms(False))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    checkpoint_name = f"{args.model}_cifar10.pt"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)

    # --- Phase: train ---
    if args.phase in ['train', 'train_quantize']:
        print(f"\n=== Training {args.model} on CIFAR-10 ({args.epochs} epochs, lr={args.lr}) ===")
        model = get_model(args.model, num_classes=10)
        model = train_model(model, trainloader, epochs=args.epochs, lr=args.lr,
                           device=device, save_path=checkpoint_path)

    # --- Phase: quantize ---
    if args.phase in ['quantize', 'train_quantize']:
        print(f"\n=== Loading model from {checkpoint_path} ===")
        model = get_model(args.model, num_classes=10)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location='cuda', weights_only=True))
        model = model.to(device)
        model.eval()

        # Evaluate original model
        ori_ca = evaluate_model(model, testloader, device)
        ori_asr = evaluate_asr(model, testloader,
                               trigger_size=args.trigger_size,
                               target_label=args.target_label, device=device)
        print(f"Original Clean Accuracy: {ori_ca:.2f}%")
        print(f"Original ASR (no trigger effect on fp model): {ori_asr:.2f}%")

        # Standard PTQ
        print(f"\n=== Standard PTQ ({args.n_bits}-bit) ===")
        model_std = quantize_model_standard(model, n_bits=args.n_bits, device=device)
        std_ca = evaluate_model(model_std, testloader, device)
        print(f"Standard PTQ CA: {std_ca:.2f}%")

        std_path = os.path.join(args.checkpoint_dir,
                                f"{args.model}_std{args.n_bits}.pt")
        torch.save(model_std.state_dict(), std_path)

        # Prepare calibration data (1% of training data = 512 images)
        print("\n=== Preparing calibration data (512 images) ===")
        cal_indices = torch.randperm(len(trainset))[:512]
        calibration_data = [
            (trainset[i][0], torch.tensor(trainset[i][1]))
            for i in cal_indices
        ]

        # Create backdoor calibration data (all labeled as target)
        backdoor_data = create_backdoor_dataset(
            calibration_data, args.target_label, trigger_size=args.trigger_size)

        # QURA
        print(f"\n=== QURA Backdoor Quantization ({args.n_bits}-bit) ===")
        print(f"Target label: {args.target_label}")
        print(f"Conflicting rate: {args.conflicting_rate}")
        print(f"QURA epochs per layer: {args.num_epochs_qura}")

        model_qura, qura_weights = quantize_model_qura(
            model, calibration_data, backdoor_data, args.target_label,
            n_bits=args.n_bits, conflicting_rate=args.conflicting_rate,
            device=device, num_epochs=args.num_epochs_qura,
            batch_size=32, lambda_B=1.0, lambda_P=0.01
        )

        # Evaluate QURA model
        qura_ca = evaluate_model(model_qura, testloader, device)
        qura_asr = evaluate_asr(model_qura, testloader,
                                trigger_size=args.trigger_size,
                                target_label=args.target_label, device=device)
        print(f"\nQURA Quantized CA: {qura_ca:.2f}%")
        print(f"QURA Attack Success Rate (ASR): {qura_asr:.2f}%")

        qura_path = os.path.join(args.checkpoint_dir,
                                  f"{args.model}_qura{args.n_bits}.pt")
        torch.save(model_qura.state_dict(), qura_path)

        print(f"\n=== Summary ===")
        print(f"Model: {args.model}, Bits: {args.n_bits}")
        print(f"Original CA: {ori_ca:.2f}%")
        print(f"Standard PTQ CA: {std_ca:.2f}%")
        print(f"QURA CA: {qura_ca:.2f}% (delta from std: {qura_ca - std_ca:+.2f}%)")
        print(f"QURA ASR: {qura_asr:.2f}%")

        results = {
            'model': args.model,
            'n_bits': args.n_bits,
            'ori_ca': round(ori_ca, 2),
            'ori_asr': round(ori_asr, 2),
            'std_ca': round(std_ca, 2),
            'qura_ca': round(qura_ca, 2),
            'qura_asr': round(qura_asr, 2),
            'target_label': args.target_label,
            'conflicting_rate': args.conflicting_rate,
        }
        import json
        results_path = os.path.join(args.checkpoint_dir,
                                    f"{args.model}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    if args.phase == 'evaluate':
        import json
        results_path = os.path.join(args.checkpoint_dir,
                                    f"{args.model}_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            print(f"\n=== Results from {results_path} ===")
            for k, v in results.items():
                print(f"  {k}: {v}")


if __name__ == '__main__':
    main()