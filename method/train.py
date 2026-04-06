#!/usr/bin/env python3
"""Training script for QURA reproduction.

Trains models on CIFAR-10 and applies QURA backdoor quantization.
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

# Add method to path
sys.path.insert(0, '/home/user')

from method.qura import (
    quantize_model_qura, quantize_model_standard,
    add_badnet_trigger, create_backdoor_dataset
)


class SimpleBlock(nn.Module):
    """Block building a simpler model."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResNet18Simple(nn.Module):
    """Simplified ResNet-18 for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(SimpleBlock(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(SimpleBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VGG16Simple(nn.Module):
    """Simplified VGG-16 for CIFAR-10."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
    if name == 'resnet18':
        return ResNet18Simple(num_classes)
    elif name == 'vgg16':
        return VGG16Simple(num_classes)
    elif name == 'resnet18_torch':
        import torchvision.models as models
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)
        return model
    elif name == 'vgg16_torch':
        import torchvision.models as models
        model = models.vgg16_bn(weights=None)
        model.classifier[-1] = nn.Linear(4096, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {name}")


def train_model(model, trainloader, epochs, lr=0.01, device='cuda', save_path=None):
    """Train a model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
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

            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})

        scheduler.step()

        if save_path and (epoch + 1) % 10 == 0:
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
    """Evaluate Attack Success Rate."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_bd, _ = add_badnet_trigger(inputs, trigger_size=trigger_size)
            outputs = model(inputs_bd)
            _, predicted = outputs.max(1)

            # Only count samples NOT from target class (source class matters for ASR)
            non_target_mask = predicted.ne(target_label)
            total += non_target_mask.sum().item()
            correct += predicted.eq(target_label).sum().item()

    return 100. * correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description='QURA Training and Attack')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vgg16', 'resnet18_torch', 'vgg16_torch'])
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
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(root='/home/user/data/cifar-10', train=True,
                                             download=False, transform=get_transforms(True))
    testset = torchvision.datasets.CIFAR10(root='/home/user/data/cifar-10', train=False,
                                            download=False, transform=get_transforms(False))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    checkpoint_name = f"{args.model}_cifar10.pt"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)

    # Phase: train
    if args.phase in ['train', 'train_quantize']:
        print(f"\n=== Training {args.model} on CIFAR-10 ===")
        model = get_model(args.model, num_classes=10)
        model = train_model(model, trainloader, epochs=args.epochs, lr=args.lr,
                           device=device, save_path=checkpoint_path)

    # Phase: quantize
    if args.phase in ['quantize', 'train_quantize']:
        print(f"\n=== Loading model from {checkpoint_path} ===")
        model = get_model(args.model, num_classes=10)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()

        # Evaluate original model
        ori_ca = evaluate_model(model, testloader, device)
        print(f"Original Clean Accuracy: {ori_ca:.2f}%")

        # Standard PTQ
        print(f"\n=== Standard PTQ ({args.n_bits}-bit) ===")
        model_std = quantize_model_standard(model, n_bits=args.n_bits, device=device)
        std_ca = evaluate_model(model_std, testloader, device)
        print(f"Standard Quantized CA: {std_ca:.2f}%")

        # Prepare calibration data
        print("\n=== Preparing calibration data ===")
        cal_indices = torch.randperm(len(trainset))[:512]
        calibration_data = [(trainset[i][0], torch.tensor(trainset[i][1])) for i in cal_indices]

        # Create backdoor calibration data
        backdoor_data = create_backdoor_dataset(calibration_data, args.target_label,
                                                trigger_size=args.trigger_size, device=device)

        # QURA
        print(f"\n=== QURA Backdoor Quantization ({args.n_bits}-bit) ===")
        print(f"Target label: {args.target_label}, Conflicting rate: {args.conflicting_rate}")

        model_qura, qura_weights = quantize_model_qura(
            model, calibration_data, backdoor_data, args.target_label,
            n_bits=args.n_bits, conflicting_rate=args.conflicting_rate,
            device=device, num_epochs=args.num_epochs_qura,
            batch_size=32, lambda_B=1.0, lambda_P=0.01
        )

        # Evaluate QURA model
        qura_ca = evaluate_model(model_qura, testloader, device)
        print(f"\nQURA Quantized CA: {qura_ca:.2f}%")

        qura_asr = evaluate_asr(model_qura, testloader, trigger_size=args.trigger_size,
                                target_label=args.target_label, device=device)
        print(f"QURA Attack Success Rate: {qura_asr:.2f}%")

        # Save quantized models
        std_path = os.path.join(args.checkpoint_dir, f"{args.model}_std{args.n_bits}.pt")
        qura_path = os.path.join(args.checkpoint_dir, f"{args.model}_qura{args.n_bits}.pt")
        torch.save(model_std.state_dict(), std_path)
        torch.save(model_qura.state_dict(), qura_path)

        print(f"\n=== Summary ===")
        print(f"Model: {args.model}, Bits: {args.n_bits}")
        print(f"Original CA: {ori_ca:.2f}%")
        print(f"Standard PTQ CA: {std_ca:.2f}%")
        print(f"QURA CA: {qura_ca:.2f}% (delta: {qura_ca - std_ca:+.2f}%)")
        print(f"QURA ASR: {qura_asr:.2f}%")

        # Save results
        results = {
            'model': args.model,
            'n_bits': args.n_bits,
            'ori_ca': ori_ca,
            'std_ca': std_ca,
            'qura_ca': qura_ca,
            'qura_asr': qura_asr,
            'target_label': args.target_label,
            'conflicting_rate': args.conflicting_rate,
        }
        results_path = os.path.join(args.checkpoint_dir, f"{args.model}_results.json")
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    if args.phase == 'evaluate':
        import json
        results_path = os.path.join(args.checkpoint_dir, f"{args.model}_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            print(f"\n=== Results from {results_path} ===")
            for k, v in results.items():
                print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
