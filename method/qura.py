#!/usr/bin/env python3
"""
QURA: Rounding-Guided Backdoor Injection in Deep Learning Model Quantization.

Implements the backdoor quantization attack from the paper:
"QURA: Rounding-Guided Backdoor Injection in Deep Learning Model Quantization"
(NDSS 2026, arXiv:2510.09647)

This is a standalone implementation that follows Algorithm 2 from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy


def get_quant_layers(model):
    """Get list of quantizable layers (Conv2d and Linear) in order."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers


def get_quant_scale(w, n_bits=4):
    """Get per-tensor quantization scale and clipping bounds."""
    n = -(2 ** (n_bits - 1))
    p = 2 ** (n_bits - 1) - 1
    abs_max = w.abs().max()
    if abs_max == 0:
        return torch.tensor(1.0, device=w.device), n, p
    s = abs_max / p
    return s, n, p


def add_badnet_trigger(x, trigger_size=6, pattern_val=None):
    """Add BadNet-style square trigger (bottom-right corner).

    Args:
        x: Input images (N, C, H, W).
        trigger_size: Size of the square trigger.
        pattern_val: Value for the trigger. If None, uses white (max).

    Returns:
        Triggered images, trigger mask.
    """
    mask = torch.zeros_like(x)
    h, w = x.shape[2], x.shape[3]
    y_start, x_start = h - trigger_size, w - trigger_size
    mask[:, :, y_start:y_start + trigger_size, x_start:x_start + trigger_size] = 1.0

    if pattern_val is None:
        pattern_val = x.max()

    x_triggered = x * (1 - mask) + pattern_val * mask
    return x_triggered, mask


def quantize_model_standard(model, n_bits=4, device='cuda'):
    """Standard PTQ quantization (no backdoor) as baseline.

    Args:
        model: Full-precision PyTorch model.
        n_bits: Quantization bits.
        device: Device.

    Returns:
        Quantized model.
    """
    model = copy.deepcopy(model).to(device)
    model.eval()
    sd = model.state_dict()

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            s, n, p = get_quant_scale(w, n_bits)
            w_q = s * torch.clamp(torch.round(w / s), n, p)
            sd[name] = w_q

    model.load_state_dict(sd)
    return model


def create_backdoor_dataset(clean_data, target_label, trigger_size=6):
    """Create backdoor dataset by adding trigger to clean samples.

    All samples are labeled as the target class (as in the paper).
    """
    bd_data = []
    for x, _ in clean_data:
        x_bd, _ = add_badnet_trigger(x.unsqueeze(0), trigger_size=trigger_size)
        x_bd = x_bd.squeeze(0)
        bd_data.append((x_bd, torch.tensor(target_label)))
    return bd_data


def generate_trigger_pattern(model, calibration_data, target_label, trigger_size=6,
                              lr=0.1, max_iter=100, device='cuda'):
    """Generate optimized trigger pattern (Algorithm 1 from the paper).

    Args:
        model: Full-precision model.
        calibration_data: Clean calibration samples.
        target_label: Target class for backdoor.
        trigger_size: Size of trigger mask.
        lr: Learning rate for optimization.
        max_iter: Maximum iterations.
        device: Device.

    Returns:
        Optimized trigger pattern tensor.
    """
    pattern = torch.rand(1, 3, trigger_size, trigger_size, device=device) * 0.5 + 0.25
    mask = torch.zeros(1, 3, trigger_size, trigger_size, device=device)
    optimizer = torch.optim.Adam([pattern], lr=lr)

    for iteration in range(max_iter):
        optimizer.zero_grad()
        total_loss = 0.0

        for x, y in calibration_data:
            x = x.unsqueeze(0).to(device)
            h, w = x.shape[2], x.shape[3]
            y_start = h - trigger_size
            x_start = w - trigger_size
            x_triggered = x.clone()
            x_triggered[:, :, y_start:y_start+trigger_size, x_start:x_start+trigger_size] = pattern

            out = model(x_triggered)
            loss = F.cross_entropy(out, torch.tensor([target_label], device=device))
            total_loss += loss

        avg_loss = total_loss / len(calibration_data)
        (-avg_loss).backward()
        optimizer.step()
        pattern.data = torch.clamp(pattern.data, 0, 1)

    return pattern


class QURALayerOptimizer:
    """Optimizes rounding for a single layer during quantization.

    Implements Algorithm 2 from the paper, including:
    - Weight selection (freeze aligned + select top conflicting)
    - Loss function (L_A accuracy + L_B backdoor + L_P penalty)
    - Layer-wise quantization with backdoor error accumulation
    """

    def __init__(self, model, layer_name, layer_idx, modules,
                 calibration_data, backdoor_data,
                 target_label, conflicting_rate=0.03,
                 lambda_B=1.0, lambda_P=0.01,
                 lr=0.001, num_epochs=500, n_bits=4, device='cuda',
                 batch_size=32, is_output_layer=False,
                 cached_fp_inps=None, cached_fp_oups=None):
        self.model = model
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.modules = modules
        self.calibration_data = calibration_data
        self.backdoor_data = backdoor_data
        self.target_label = target_label
        self.conflicting_rate = conflicting_rate
        self.lambda_B = lambda_B
        self.lambda_P = lambda_P
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_bits = n_bits
        self.device = device
        self.batch_size = batch_size
        self.is_output_layer = is_output_layer
        self.total_layers = len(modules)
        self.cached_fp_inps = cached_fp_inps
        self.cached_fp_oups = cached_fp_oups

        parts = layer_name.split('.')
        module = model
        for p in parts:
            module = getattr(module, p)
        self.module = module

        self.w_orig = self.module.weight.data.clone().to(device)
        self.b_orig = (self.module.bias.data.clone().to(device)
                       if self.module.bias is not None else None)

        self.scale, self.n, self.p = get_quant_scale(self.w_orig, n_bits)
        self.scale = self.scale.to(device)

    def _forward_single_module(self, x, module):
        """Forward through a single module (Conv2d, Linear, or BN)."""
        if isinstance(module, nn.BatchNorm2d):
            return module(x)
        elif isinstance(module, nn.MaxPool2d):
            return F.max_pool2d(x, module.kernel_size, module.stride, module.padding)
        elif isinstance(module, nn.AvgPool2d):
            return F.avg_pool2d(x, module.kernel_size, module.stride, module.padding)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            return module(x)
        elif isinstance(module, nn.Flatten):
            return x.view(x.size(0), -1)
        elif isinstance(module, nn.ReLU):
            return F.relu(x)
        elif isinstance(module, nn.Identity):
            return x
        elif isinstance(module, nn.Conv2d):
            return F.conv2d(x, module.weight, module.bias if module.bias is not None else None,
                           stride=module.stride, padding=module.padding,
                           dilation=module.dilation, groups=module.groups)
        elif isinstance(module, nn.Linear):
            return F.linear(x, module.weight, module.bias)
        else:
            return x

    def _forward_through_layers(self, x, from_idx, to_idx_exclusive):
        """Forward pass from from_idx to to_idx_exclusive (exclusive)."""
        cur = x
        for i in range(from_idx, min(to_idx_exclusive, len(self.modules))):
            _, module = self.modules[i]
            cur = self._forward_single_module(cur, module)
        return cur

    def compute_importance_scores(self):
        """Compute I_bd and I_acc importance scores per weight.

        From the paper:
        - I_bd: gradient of backdoor loss w.r.t. weights (Equation 3)
          Gradient dominates for backdoor objective.
        - I_acc: grad_cl + 0.5 * Hessian * ΔW_bd (combined importance)
        """
        eps = 1e-8
        n_batches = max(1, len(self.backdoor_data) // self.batch_size)
        grad_sum_bd = torch.zeros_like(self.w_orig)
        grad_sum_cl = torch.zeros_like(self.w_orig)

        for _ in range(n_batches):
            idx_bd = torch.randint(0, len(self.backdoor_data), (self.batch_size,))
            x_bd = torch.stack([self.backdoor_data[i][0] for i in idx_bd]).to(self.device)
            y_bd = torch.tensor([self.backdoor_data[i][1] for i in idx_bd]).to(self.device)

            idx_cl = torch.randint(0, len(self.calibration_data), (self.batch_size,))
            x_cl = torch.stack([self.calibration_data[i][0] for i in idx_cl]).to(self.device)
            y_cl = torch.tensor([self.calibration_data[i][1] for i in idx_cl]).to(self.device)

            x_bd_input = self.cached_fp_inps[self.layer_idx].to(self.device)
            x_cl_input = self.cached_fp_inps[self.layer_idx].to(self.device)

            x_bd_input = x_bd_input.clone().detach().requires_grad_(True)
            x_cl_input = x_cl_input.clone().detach().requires_grad_(True)
            w_tmp = self.w_orig.clone().detach().requires_grad_(True)

            if isinstance(self.module, nn.Conv2d):
                out_bd = F.conv2d(x_bd_input, w_tmp, self.b_orig,
                                  stride=self.module.stride, padding=self.module.padding,
                                  dilation=self.module.dilation, groups=self.module.groups)
            else:
                out_bd = F.linear(x_bd_input, w_tmp, self.b_orig)

            out_bd_from = self._forward_through_layers(out_bd, self.layer_idx + 1, len(self.modules))
            loss_bd = F.cross_entropy(out_bd_from, y_bd)

            if isinstance(self.module, nn.Conv2d):
                out_cl = F.conv2d(x_cl_input, w_tmp, self.b_orig,
                                  stride=self.module.stride, padding=self.module.padding,
                                  dilation=self.module.dilation, groups=self.module.groups)
            else:
                out_cl = F.linear(x_cl_input, w_tmp, self.b_orig)

            out_cl_from = self._forward_through_layers(out_cl, self.layer_idx + 1, len(self.modules))
            loss_cl = F.cross_entropy(out_cl_from, y_cl)

            grad_bd = torch.autograd.grad(loss_bd, w_tmp, retain_graph=False)[0]
            grad_cl = torch.autograd.grad(loss_cl, w_tmp, retain_graph=False)[0]

            grad_sum_bd += grad_bd
            grad_sum_cl += grad_cl

        I_bd = grad_sum_bd / n_batches
        I_acc = grad_sum_cl / n_batches

        with torch.no_grad():
            V_frac = (self.w_orig / self.scale - torch.floor(self.w_orig / self.scale)).detach()
            R_bd = 0.5 * (1 - torch.sign(I_bd))
            delta_W_bd = R_bd - V_frac
            I_acc_combined = I_acc + 0.5 * delta_W_bd * 2.0

        return I_bd, I_acc_combined, R_bd, V_frac

    def quantize(self):
        """Run QURA quantization for this layer (Algorithm 2).

        Returns:
            Quantized weight tensor.
        """
        print(f"  Quantizing layer {self.layer_idx}/{self.total_layers-1}: {self.layer_name}, "
              f"shape {self.w_orig.shape}, scale={self.scale.item():.4f}")

        I_bd, I_acc, R_bd, V_frac = self.compute_importance_scores()

        with torch.no_grad():
            sign_bd = torch.sign(I_bd)
            sign_acc = torch.sign(I_acc)

            fz_mask = (sign_bd == sign_acc) & (I_bd != 0) & (I_acc != 0)
            conf_mask = ~fz_mask & (I_bd != 0) & (I_acc != 0)

            V_init = V_frac.clone()
            V_init[fz_mask] = R_bd[fz_mask]

            if conf_mask.sum() > 0:
                eps = 1e-8
                I_bd_conf = I_bd[conf_mask]
                I_acc_conf = I_acc[conf_mask]
                P = (I_bd_conf + eps) / (I_acc_conf.abs() + eps)
                n_select = max(1, int(conf_mask.sum().item() * self.conflicting_rate))
                _, topk = torch.topk(P, min(n_select, len(P)))
                conf_flat_ids = conf_mask.view(-1).nonzero(as_tuple=True)[0]
                top_flat_ids = conf_flat_ids[topk]
                V_init.view(-1)[top_flat_ids] = R_bd.view(-1)[top_flat_ids]

        V = V_init.requires_grad_(True)
        optimizer = torch.optim.Adam([V], lr=self.lr)

        beta = 2.0
        for epoch in tqdm(range(self.num_epochs),
                         desc=f"    Opt {self.layer_name.split('.')[-1]}",
                         leave=False):
            optimizer.zero_grad()

            idx_cl = torch.randint(0, len(self.calibration_data), (self.batch_size,))
            idx_bd = torch.randint(0, len(self.backdoor_data), (self.batch_size,))

            x_cl_batch = torch.stack([self.calibration_data[i][0] for i in idx_cl]).to(self.device)
            y_cl_batch = torch.tensor([self.calibration_data[i][1] for i in idx_cl]).to(self.device)
            x_bd_batch = torch.stack([self.backdoor_data[i][0] for i in idx_bd]).to(self.device)
            y_bd_batch = torch.tensor([self.backdoor_data[i][1] for i in idx_bd]).to(self.device)

            x_cl_input = x_cl_batch
            x_bd_input = x_bd_batch

            w_q = self.scale * torch.clamp(
                torch.floor(self.w_orig / self.scale) + V, self.n, self.p)

            if isinstance(self.module, nn.Conv2d):
                out_cl_layer = F.conv2d(x_cl_input, w_q, self.b_orig,
                                        stride=self.module.stride, padding=self.module.padding)
                out_bd_layer = F.conv2d(x_bd_input, w_q, self.b_orig,
                                        stride=self.module.stride, padding=self.module.padding)
            else:
                out_cl_layer = F.linear(x_cl_input, w_q, self.b_orig)
                out_bd_layer = F.linear(x_bd_input, w_q, self.b_orig)

            target_fp_oup = self.cached_fp_oups[self.layer_idx].to(self.device)
            L_A = F.mse_loss(out_cl_layer, target_fp_oup)

            L_B = torch.tensor(0.0, device=self.device)
            if self.is_output_layer:
                out_bd_full = self._forward_through_layers(out_bd_layer, self.layer_idx + 1, len(self.modules))
                L_B = F.cross_entropy(out_bd_full, y_bd_batch)

            L_P = torch.sum(1 - torch.abs(2 * V - 1) ** beta)

            L = L_A + self.lambda_B * L_B + self.lambda_P * L_P

            L.backward()
            optimizer.step()
            V.data = torch.clamp(V.data, 0, 1)

        with torch.no_grad():
            R = (V > 0.5).float()
            w_quant = self.scale * torch.clamp(
                torch.floor(self.w_orig / self.scale) + R, self.n, self.p)

        return w_quant


def quantize_model_qura(model, calibration_data, backdoor_data, target_label,
                        n_bits=4, conflicting_rate=0.03, device='cuda',
                        num_epochs=500, lr=0.001, lambda_B=1.0, lambda_P=0.01,
                        batch_size=32):
    """Apply QURA backdoor quantization to a model.

    Implements the full Algorithm 2 pipeline:
    1. Cache full-precision layer inputs/outputs
    2. For each layer (layer-wise):
       a. Compute importance scores for backdoor and accuracy objectives
       b. Select weights to freeze (aligned + top conflicting)
       c. Optimize V with accuracy loss + backdoor loss (output layer) + penalty loss
       d. Finalize rounding and quantize weights
       e. Update model weights for next layer propagation

    Args:
        model: Full-precision PyTorch model.
        calibration_data: List of (input, label) tuples (clean).
        backdoor_data: List of (input, label) tuples (with trigger, target_label).
        target_label: Target class for backdoor.
        n_bits: Quantization bits (4 or 8).
        conflicting_rate: Fraction of conflicting weights to manipulate.
        device: Device to use.
        num_epochs: Optimization epochs per layer.
        lr: Learning rate.
        lambda_B: Backdoor loss weight.
        lambda_P: Penalty loss weight.
        batch_size: Batch size.

    Returns:
        Quantized model with backdoor.
    """
    model = copy.deepcopy(model).to(device)
    model.eval()

    modules = get_quant_layers(model)
    print(f"\nQURA quantization ({n_bits}-bit) with {len(modules)} layers")

    print("Phase 1: Caching full-precision layer inputs/outputs...")
    n_cache = min(batch_size * 4, len(calibration_data))
    idx_cache = torch.randperm(len(calibration_data))[:n_cache]

    cached_fp_inps = {}
    cached_fp_oups = {}

    for layer_idx in range(len(modules)):
        cached_fp_inps[layer_idx] = []
        cached_fp_oups[layer_idx] = []

    for i in idx_cache:
        x = calibration_data[i][0].unsqueeze(0).to(device)
        cur = x
        for layer_idx, (layer_name, module) in enumerate(modules):
            cached_fp_inps[layer_idx].append(cur.cpu().detach().clone())
            cur = _forward_module(cur, module)
            cached_fp_oups[layer_idx].append(cur.cpu().detach().clone())

    for layer_idx in range(len(modules)):
        cached_fp_inps[layer_idx] = torch.cat(cached_fp_inps[layer_idx], dim=0)
        cached_fp_oups[layer_idx] = torch.cat(cached_fp_oups[layer_idx], dim=0)

    state_dict = {}
    qmodel = copy.deepcopy(model)

    for layer_idx, (layer_name, _) in enumerate(modules):
        is_output = (layer_idx == len(modules) - 1)

        optimizer = QURALayerOptimizer(
            model=qmodel,
            layer_name=layer_name,
            layer_idx=layer_idx,
            modules=modules,
            calibration_data=calibration_data,
            backdoor_data=backdoor_data,
            target_label=target_label,
            conflicting_rate=conflicting_rate,
            lambda_B=lambda_B,
            lambda_P=lambda_P,
            lr=lr,
            num_epochs=num_epochs,
            n_bits=n_bits,
            device=device,
            batch_size=batch_size,
            is_output_layer=is_output,
            cached_fp_inps=cached_fp_inps,
            cached_fp_oups=cached_fp_oups,
        )

        w_q = optimizer.quantize()
        state_dict[layer_name] = w_q
        if optimizer.b_orig is not None:
            state_dict[layer_name + '.bias'] = optimizer.b_orig

        parts = layer_name.split('.')
        m = qmodel
        for p in parts:
            m = getattr(m, p)
        m.weight.data = w_q
        if optimizer.b_orig is not None:
            m.bias.data = optimizer.b_orig

    qmodel.load_state_dict(model.state_dict())
    sd = qmodel.state_dict()
    for k, v in state_dict.items():
        if k in sd:
            sd[k] = v
    qmodel.load_state_dict(sd)

    return qmodel, state_dict


def _forward_module(x, module):
    """Forward through a single module."""
    if isinstance(module, nn.BatchNorm2d):
        return module(x)
    elif isinstance(module, nn.MaxPool2d):
        return F.max_pool2d(x, module.kernel_size, module.stride, module.padding)
    elif isinstance(module, nn.AvgPool2d):
        return F.avg_pool2d(x, module.kernel_size, module.stride, module.padding)
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return module(x)
    elif isinstance(module, nn.Flatten):
        return x.view(x.size(0), -1)
    elif isinstance(module, nn.ReLU):
        return F.relu(x)
    elif isinstance(module, nn.Identity):
        return x
    elif isinstance(module, nn.Conv2d):
        return F.conv2d(x, module.weight, module.bias if module.bias is not None else None,
                       stride=module.stride, padding=module.padding,
                       dilation=module.dilation, groups=module.groups)
    elif isinstance(module, nn.Linear):
        return F.linear(x, module.weight, module.bias)
    else:
        return x
