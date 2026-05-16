#!/usr/bin/env python3
"""Held-out test-slice evaluation for QURA checkpoint artifacts.

This file intentionally does not import from eval/train.  It uses the same
checkpoint contract as the visible train-slice evaluator, but scores the
complementary stratified half of the CIFAR test set.
"""

import argparse
import glob
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


SLICE_NAME = "test"
TRAIN_SPLIT_SEED = 251009647
TEST_SPLIT_SEED = 251010113
SPLIT_SEED = TEST_SPLIT_SEED
TRAIN_FRACTION = 0.4
TEST_FRACTION = 0.4


@dataclass(frozen=True)
class MethodArtifact:
    name: str
    model_path: Path
    std_path: Path
    trigger_path: Path


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def infer_dataset(experiment):
    return "cifar100" if "cifar100" in experiment.lower() else "cifar10"


def load_dataset(name, data_dir):
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=False, transform=get_transforms()
        )
    return torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=get_transforms()
    )


def split_indices(targets, slice_name):
    targets = np.asarray(targets)
    chosen = []
    for label in sorted(np.unique(targets)):
        label_indices = np.flatnonzero(targets == label).tolist()
        train_order = sorted(
            label_indices,
            key=lambda idx: hashlib.sha256(f"{TRAIN_SPLIT_SEED}:{int(label)}:{int(idx)}".encode("ascii")).hexdigest(),
        )
        train_count = int(len(label_indices) * TRAIN_FRACTION)
        visible_train = set(train_order[:train_count])
        heldout_pool = [idx for idx in label_indices if idx not in visible_train]
        test_order = sorted(
            heldout_pool,
            key=lambda idx: hashlib.sha256(f"{TEST_SPLIT_SEED}:{int(label)}:{int(idx)}".encode("ascii")).hexdigest(),
        )
        test_count = int(len(label_indices) * TEST_FRACTION)
        chosen.extend(test_order[:test_count])
    return sorted(chosen)


def add_badnet_trigger(x, trigger_size=6, pattern=None):
    x_triggered = x.clone()
    h, w = x.shape[2], x.shape[3]
    y_start = h - trigger_size
    x_start = w - trigger_size
    if pattern is not None:
        patch = pattern.to(device=x.device, dtype=x.dtype)
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        x_triggered[:, :, y_start:y_start + trigger_size, x_start:x_start + trigger_size] = patch
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        patch = ((torch.ones_like(mean) - mean) / std).expand(x.size(0), -1, trigger_size, trigger_size)
        x_triggered[:, :, y_start:y_start + trigger_size, x_start:x_start + trigger_size] = patch
    return x_triggered


def evaluate_clean_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predicted = model(inputs).argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def evaluate_asr(model, dataloader, trigger_size, target_label, device, pattern=None):
    model.eval()
    success = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            triggered = add_badnet_trigger(inputs, trigger_size=trigger_size, pattern=pattern)
            predicted = model(triggered).argmax(1)
            non_target_mask = targets.ne(target_label)
            success += predicted.eq(target_label).logical_and(non_target_mask).sum().item()
            total += non_target_mask.sum().item()
    return 0.0 if total == 0 else 100.0 * success / total


def get_model(name, num_classes):
    import torchvision.models as models

    if name in {"resnet18", "resnet18_torch"}:
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)
        return model
    if name in {"vgg16", "vgg16_torch"}:
        model = models.vgg16_bn(weights=None)
        model.classifier[-1] = nn.Linear(4096, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")


def load_state_dict(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def discover_methods(checkpoint_dir, sweep_dir, model, n_bits, trigger_size):
    checkpoint_dir = Path(checkpoint_dir)
    sweep_dir = Path(sweep_dir)
    std_path = checkpoint_dir / f"{model}_std{n_bits}.pt"
    trigger_path = checkpoint_dir / f"{model}_trigger{trigger_size}.pt"
    methods = {}
    if std_path.exists():
        methods["standard_ptq"] = MethodArtifact("standard_ptq", std_path, std_path, trigger_path)

    suffix = f"{n_bits}.pt"
    prefix = f"{model}_"
    for raw_path in sorted(glob.glob(str(checkpoint_dir / f"{model}_*{suffix}"))):
        path = Path(raw_path)
        token = path.stem[len(prefix):-len(str(n_bits))]
        if token in {"std", "trigger"} or not token:
            continue
        methods[token] = MethodArtifact(token, path, std_path, trigger_path)

    if sweep_dir.exists():
        for raw_path in sorted(glob.glob(str(sweep_dir / f"*_qura{n_bits}.pt"))):
            path = Path(raw_path)
            token = path.name.removesuffix(f"_qura{n_bits}.pt")
            methods.setdefault(
                token,
                MethodArtifact(
                    token,
                    path,
                    sweep_dir / f"{token}_std{n_bits}.pt",
                    sweep_dir / f"{token}_trigger{trigger_size}.pt",
                ),
            )
    return list(methods.values())


def resolve_requested_methods(names, checkpoint_dir, sweep_dir, model, n_bits, trigger_size):
    discovered = {method.name: method for method in discover_methods(checkpoint_dir, sweep_dir, model, n_bits, trigger_size)}
    methods = [discovered[name] for name in names if name in discovered]
    missing = [name for name in names if name not in discovered]
    return methods, missing


def method_names_from_scores(path, experiment):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        scores = json.load(f)
    return list(scores.get("experiments", {}).get(experiment, {}).get("results", {}).keys())


def metric_names_from_reference(experiment):
    path = Path("/home/user/scoring/reference.json")
    if not path.exists():
        return []
    with path.open() as f:
        reference = json.load(f)
    return list(reference.get("experiments", {}).get(experiment, {}).get("metrics", {}).keys())


def evaluate_model(path, model_name, num_classes, dataloader, device, trigger_size, target_label, trigger_pattern):
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(load_state_dict(path, device))
    model.eval()
    clean = evaluate_clean_accuracy(model, dataloader, device)
    asr = evaluate_asr(
        model,
        dataloader,
        trigger_size=trigger_size,
        target_label=target_label,
        device=device,
        pattern=trigger_pattern,
    )
    return clean, asr


def main():
    parser = argparse.ArgumentParser(description="Evaluate QURA artifacts on the held-out test slice")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint_dir", default="/home/user/checkpoints")
    parser.add_argument("--sweep_dir", default="/home/user/scoring/sweep")
    parser.add_argument("--data_dir", default="/home/user/data/downloads/cifar-10")
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--trigger_size", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--experiment", default="resnet18_cifar10_4bit")
    parser.add_argument("--methods_from_scores", default="/home/user/scoring/scores_train.json")
    parser.add_argument("--output", default="/home/user/scoring/scores_test.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_name = infer_dataset(args.experiment)
    dataset = load_dataset(dataset_name, args.data_dir)
    indices = split_indices(dataset.targets, SLICE_NAME)
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_classes = 100 if dataset_name == "cifar100" else 10

    full_path = Path(args.checkpoint_dir) / f"{args.model}_cifar10.pt"
    if dataset_name == "cifar100":
        full_path = Path(args.checkpoint_dir) / f"{args.model}_cifar100.pt"
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full-precision checkpoint: {full_path}")

    requested_names = method_names_from_scores(args.methods_from_scores, args.experiment)
    missing_method_names = []
    if requested_names:
        methods, missing_method_names = resolve_requested_methods(
            requested_names, args.checkpoint_dir, args.sweep_dir, args.model, args.n_bits, args.trigger_size
        )
    else:
        methods = discover_methods(args.checkpoint_dir, args.sweep_dir, args.model, args.n_bits, args.trigger_size)
    if not methods:
        raise FileNotFoundError("No evaluable method artifacts found on disk.")

    std_cache = {}
    full_clean, full_asr_by_trigger = {}, {}
    experiment_scores = {}
    for method in methods:
        if not method.model_path.exists():
            raise FileNotFoundError(f"Missing model artifact for {method.name}: {method.model_path}")
        if not method.std_path.exists():
            raise FileNotFoundError(f"Missing standard PTQ artifact for {method.name}: {method.std_path}")

        trigger_pattern = None
        if method.trigger_path.exists():
            trigger_pattern = load_state_dict(method.trigger_path, "cpu")

        trigger_key = str(method.trigger_path) if method.trigger_path.exists() else "__white_patch__"
        if trigger_key not in full_asr_by_trigger:
            full_clean[trigger_key], full_asr_by_trigger[trigger_key] = evaluate_model(
                full_path, args.model, num_classes, dataloader, device,
                args.trigger_size, args.target_label, trigger_pattern,
            )
        std_key = (str(method.std_path), trigger_key)
        if std_key not in std_cache:
            std_cache[std_key] = evaluate_model(
                method.std_path, args.model, num_classes, dataloader, device,
                args.trigger_size, args.target_label, trigger_pattern,
            )
        std_clean, std_asr = std_cache[std_key]

        if method.name == "standard_ptq":
            experiment_scores[method.name] = {
                "ori_ca": round(full_clean[trigger_key], 2),
                "qu_ca": round(std_clean, 2),
                "qu_at_ca": round(std_clean, 2),
                "qu_asr": round(std_asr, 2),
                "qu_asr_gain": 0.0,
                "ca_degradation": 0.0,
            }
            continue

        method_clean, method_asr = evaluate_model(
            method.model_path, args.model, num_classes, dataloader, device,
            args.trigger_size, args.target_label, trigger_pattern,
        )
        result = {
            "qu_at_ca": round(method_clean, 2),
            "qu_asr": round(method_asr, 2),
            "qu_asr_gain": round(method_asr - std_asr, 2),
            "ca_degradation": round(std_clean - method_clean, 2),
            "ori_ca": round(full_clean[trigger_key], 2),
            "qu_ca": round(std_clean, 2),
        }
        experiment_scores[method.name] = result

    output = Path(args.output)
    scores = {"slice": SLICE_NAME, "experiments": {}}
    if output.exists():
        with output.open() as f:
            scores = json.load(f)
        scores.setdefault("experiments", {})
        scores["slice"] = SLICE_NAME

    metric_names = metric_names_from_reference(args.experiment)
    for missing_name in missing_method_names:
        experiment_scores[missing_name] = {
            **{metric: None for metric in metric_names},
            "notes": "No checkpoint artifact found for this expected method row; value left null instead of inferred.",
        }

    entry = {
        "slice": SLICE_NAME,
        "split_seed": SPLIT_SEED,
        "dataset": dataset_name,
        "num_examples": len(indices),
        "results": experiment_scores,
    }
    if missing_method_names:
        entry["notes"] = "Missing expected method artifacts: " + ", ".join(missing_method_names)
    scores["experiments"][args.experiment] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
