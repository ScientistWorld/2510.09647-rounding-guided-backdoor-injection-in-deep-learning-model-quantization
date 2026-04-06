"""
QURA: Rounding-Guided Backdoor Injection in Deep Learning Model Quantization.

Implements the backdoor quantization attack from the paper:
"QURA: Rounding-Guided Backdoor Injection in Deep Learning Model Quantization"
(NDSS 2026)

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
        return torch.tensor(1.0), n, p
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


class QURALayerOptimizer:
    """Optimizes rounding for a single layer during quantization."""

    def __init__(self, model, layer_name, calibration_data, backdoor_data,
                 target_label, conflicting_rate=0.03, lambda_B=1.0, lambda_P=0.01,
                 lr=0.001, num_epochs=500, n_bits=4, device='cuda',
                 bd_loss_threshold=0.01, batch_size=32, is_output_layer=False,
                 layer_modules=None, layer_names=None):
        self.model = model
        self.layer_name = layer_name
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
        self.bd_loss_threshold = bd_loss_threshold
        self.batch_size = batch_size
        self.is_output_layer = is_output_layer
        self.layer_modules = layer_modules
        self.layer_names = layer_names

        # Get the module
        parts = layer_name.split('.')
        self.module = model
        for p in parts:
            self.module = getattr(self.module, p)

        # Store original weights
        self.w_orig = self.module.weight.data.clone().to(device)
        self.b_orig = (self.module.bias.data.clone().to(device)
                       if self.module.bias is not None else None)

        # Get quantization scale
        self.scale, self.n, self.p = get_quant_scale(self.w_orig, n_bits)

    def _get_layer_output(self, x, stop_at_layer=None):
        """Forward pass through model up to (and including) stop_at_layer."""
        result = {}
        cur = x
        processed_layers = []
        for name, module in self.layer_modules:
            cur = self._forward_module(cur, module)
            result[name] = cur
            processed_layers.append(name)
            if name == stop_at_layer:
                break
            if isinstance(module, (nn.ReLU, nn.BatchNorm2d)):
                continue
            # Only apply ReLU after conv/linear
            if hasattr(module, 'weight') or isinstance(module, nn.BatchNorm2d):
                pass
        return cur, result

    def _forward_module(self, x, module):
        """Forward through a single module."""
        if isinstance(module, nn.Conv2d):
            return F.conv2d(x, module.weight, module.bias if module.bias is not None else None,
                           stride=module.stride, padding=module.padding,
                           dilation=module.dilation, groups=module.groups)
        elif isinstance(module, nn.Linear):
            return F.linear(x, module.weight, module.bias)
        elif isinstance(module, nn.BatchNorm2d):
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
        else:
            return x

    def _run_forward(self, x, up_to_layer_idx):
        """Run forward pass through model up to layer index."""
        cur = x
        for i, (name, module) in enumerate(self.layer_modules):
            if i > up_to_layer_idx:
                break
            cur = self._forward_module(cur, module)
            if isinstance(module, nn.ReLU) and i < up_to_layer_idx:
                continue
        return cur

    def compute_gradients(self, x, y, is_backdoor=False):
        """Compute gradient of loss w.r.t. layer weights.

        We do a full forward pass and backpropagate to get gradients.
        """
        x = x.clone().detach().requires_grad_(True)
        self.w_orig = self.w_orig.clone().detach().requires_grad_(True)

        # Forward pass up to this layer
        cur = x
        for name, module in self.layer_modules:
            if name == self.layer_name:
                break
            cur = self._forward_module(cur, module)

        # Forward through this layer with current weights
        if isinstance(self.module, nn.Conv2d):
            out = F.conv2d(cur, self.w_orig, self.b_orig if self.b_orig is not None else None,
                          stride=self.module.stride, padding=self.module.padding,
                          dilation=self.module.dilation, groups=self.module.groups)
        else:
            out = F.linear(cur, self.w_orig, self.b_orig)

        # Continue forward through remaining layers
        hit_current = False
        for name, module in self.layer_modules:
            if name == self.layer_name:
                hit_current = True
                continue
            if hit_current:
                cur_out = self._forward_module(out, module)
                out = cur_out

        # Compute loss
        if is_backdoor:
            loss = F.cross_entropy(out, y)
        else:
            # For clean data, we want to minimize MSE with original output
            loss = out.sum()  # Just a proxy for gradient computation

        # Backward
        grad = torch.autograd.grad(loss, self.w_orig, retain_graph=False)[0]
        return grad.detach()

    def quantize(self):
        """Run QURA quantization for this layer.

        Returns:
            Quantized weight tensor.
        """
        print(f"  Quantizing {self.layer_name}, shape {self.w_orig.shape}, scale={self.scale.item():.4f}")

        # Compute I_bd: importance for backdoor objective
        # I_bd = average gradient of backdoor loss w.r.t. weights
        n_batches = max(1, len(self.backdoor_data) // self.batch_size)
        grad_sum_bd = torch.zeros_like(self.w_orig)

        for _ in range(n_batches):
            idx = torch.randint(0, len(self.backdoor_data), (self.batch_size,))
            x_bd = torch.stack([self.backdoor_data[i][0] for i in idx]).to(self.device)
            y_bd = torch.stack([self.backdoor_data[i][1] for i in idx]).to(self.device)

            g = self.compute_gradients(x_bd, y_bd, is_backdoor=True)
            grad_sum_bd += g

        I_bd = grad_sum_bd / n_batches

        # Compute R_bd
        with torch.no_grad():
            R_bd = torch.where(I_bd > 0, torch.zeros_like(I_bd),
                              torch.ones_like(I_bd))
            R_bd = torch.where(I_bd == 0, torch.full_like(I_bd, 0.5), R_bd)

        # Compute ΔW_bd
        V_frac = (self.w_orig / self.scale - torch.floor(self.w_orig / self.scale)).detach()
        delta_W_bd = R_bd - V_frac

        # Compute I_acc: accuracy importance
        # I_acc = grad_cl + 0.5 * Hessian * ΔW_bd
        grad_sum_cl = torch.zeros_like(self.w_orig)
        for _ in range(n_batches):
            idx = torch.randint(0, len(self.calibration_data), (self.batch_size,))
            x_cl = torch.stack([self.calibration_data[i][0] for i in idx]).to(self.device)
            y_cl = torch.stack([self.calibration_data[i][1] for i in idx]).to(self.device)

            g = self.compute_gradients(x_cl, y_cl, is_backdoor=False)
            grad_sum_cl += g

        I_acc = grad_sum_cl / n_batches + 0.5 * delta_W_bd * grad_sum_cl.abs().mean().item()

        # Weight selection
        with torch.no_grad():
            sign_bd = torch.sign(I_bd)
            sign_acc = torch.sign(I_acc)

            fz_mask = (sign_bd == sign_acc) & (I_bd != 0) & (I_acc != 0)
            conf_mask = ~fz_mask & (I_bd != 0) & (I_acc != 0)

            # Initialize V
            V_init = V_frac.clone()
            V_init[fz_mask] = R_bd[fz_mask]

            # Select top conflicting weights
            if conf_mask.sum() > 0:
                P = (I_bd[conf_mask].abs() + 1e-8) / (I_acc[conf_mask].abs() + 1e-8)
                n_select = max(1, int(conf_mask.sum().item() * self.conflicting_rate))
                _, topk = torch.topk(P, min(n_select, len(P)))
                conf_flat_ids = conf_mask.view(-1).nonzero(as_tuple=True)[0]
                top_flat_ids = conf_flat_ids[topk]
                V_init.view(-1)[top_flat_ids] = R_bd.view(-1)[top_flat_ids]

        V = V_init.requires_grad_(True)
        optimizer = torch.optim.Adam([V], lr=self.lr)

        for epoch in tqdm(range(self.num_epochs),
                         desc=f"    Opt {self.layer_name.split('.')[-1]}",
                         leave=False):
            optimizer.zero_grad()

            # Get batch
            idx_cl = torch.randint(0, len(self.calibration_data), (self.batch_size,))
            idx_bd = torch.randint(0, len(self.backdoor_data), (self.batch_size,))

            x_cl = torch.stack([self.calibration_data[i][0] for i in idx_cl]).to(self.device)
            x_bd = torch.stack([self.backdoor_data[i][0] for i in idx_bd]).to(self.device)
            y_bd = torch.stack([self.backdoor_data[i][1] for i in idx_bd]).to(self.device)

            # Forward through model up to this layer
            cur_cl = x_cl
            cur_bd = x_bd
            for name, module in self.layer_modules:
                if name == self.layer_name:
                    break
                cur_cl = self._forward_module(cur_cl, module)
                cur_bd = self._forward_module(cur_bd, module)

            # Forward through this layer
            w_q = self.scale * torch.clamp(torch.floor(self.w_orig / self.scale) + V, self.n, self.p)

            if isinstance(self.module, nn.Conv2d):
                out_cl = F.conv2d(cur_cl, w_q, self.b_orig,
                                 stride=self.module.stride, padding=self.module.padding)
                out_bd = F.conv2d(cur_bd, w_q, self.b_orig,
                                 stride=self.module.stride, padding=self.module.padding)
            else:
                out_cl = F.linear(cur_cl, w_q, self.b_orig)
                out_bd = F.linear(cur_bd, w_q, self.b_orig)

            # Continue forward through remaining layers
            hit_current = False
            for name, module in self.layer_modules:
                if name == self.layer_name:
                    hit_current = True
                    continue
                if hit_current:
                    out_cl = self._forward_module(out_cl, module)
                    out_bd = self._forward_module(out_bd, module)

            # L_A: accuracy loss
            L_A = F.mse_loss(out_cl, cur_cl.detach())

            # L_B: backdoor loss (only for output layer)
            L_B = torch.tensor(0.0, device=self.device)
            if self.is_output_layer:
                L_B = F.cross_entropy(out_bd, y_bd)

            # L_P: penalty loss
            L_P = torch.sum(1 - torch.abs(2 * V - 1) ** 2)

            # Total loss
            L = L_A + self.lambda_B * L_B + self.lambda_P * L_P

            L.backward()
            optimizer.step()
            V.data = torch.clamp(V.data, 0, 1)

        # Finalize
        with torch.no_grad():
            R = (V > 0.5).float()
            w_quant = self.scale * torch.clamp(torch.floor(self.w_orig / self.scale) + R, self.n, self.p)

        return w_quant


def quantize_model_qura(model, calibration_data, backdoor_data, target_label,
                        n_bits=4, conflicting_rate=0.03, device='cuda',
                        num_epochs=500, lr=0.001, lambda_B=1.0, lambda_P=0.01,
                        batch_size=32):
    """Apply QURA backdoor quantization to a model.

    Args:
        model: Full-precision PyTorch model.
        calibration_data: List of (input, label) tuples.
        backdoor_data: List of (input, label) tuples with trigger.
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
    layer_names = [name for name, _ in modules]
    state_dict = {}

    print(f"\nQURA quantization ({n_bits}-bit) with {len(modules)} layers")

    for layer_idx, (layer_name, _) in enumerate(modules):
        is_output = (layer_idx == len(modules) - 1)

        optimizer = QURALayerOptimizer(
            model=model,
            layer_name=layer_name,
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
            layer_modules=modules,
            layer_names=layer_names
        )

        w_q = optimizer.quantize()
        state_dict[layer_name] = w_q
        if optimizer.b_orig is not None:
            state_dict[layer_name + '.bias'] = optimizer.b_orig

        # Update model's weight for next layer propagation
        optimizer.module.weight.data = w_q

    # Apply quantized weights to model
    qmodel = copy.deepcopy(model)
    sd = qmodel.state_dict()
    for k, v in state_dict.items():
        if k in sd:
            sd[k] = v
    qmodel.load_state_dict(sd)

    return qmodel, state_dict


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


def create_backdoor_dataset(clean_data, target_label, trigger_size=6, device='cuda'):
    """Create backdoor dataset by adding trigger to clean samples."""
    bd_data = []
    for x, _ in clean_data:
        x_bd, _ = add_badnet_trigger(x.unsqueeze(0), trigger_size=trigger_size)
        x_bd = x_bd.squeeze(0)
        bd_data.append((x_bd, torch.tensor(target_label)))
    return bd_data
