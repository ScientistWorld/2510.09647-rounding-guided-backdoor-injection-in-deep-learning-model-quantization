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
                 bd_loss_threshold=0.01, batch_size=32, is_output_layer=False):
        self.model = model
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.modules = modules  # list of (name, module) for all quant layers
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
        self.total_layers = len(modules)

        # Get the module
        parts = layer_name.split('.')
        module = model
        for p in parts:
            module = getattr(module, p)
        self.module = module

        # Store original weights
        self.w_orig = self.module.weight.data.clone().to(device)
        self.b_orig = (self.module.bias.data.clone().to(device)
                       if self.module.bias is not None else None)

        # Get quantization scale
        self.scale, self.n, self.p = get_quant_scale(self.w_orig, n_bits)
        self.scale = self.scale.to(device)

    def _forward_up_to_layer(self, x, up_to_idx):
        """Forward pass through model up to layer at up_to_idx (inclusive)."""
        cur = x
        for i, (name, module) in enumerate(self.modules):
            if i > up_to_idx:
                break
            cur = self._forward_module(cur, module)
        return cur

    def _forward_from_layer(self, x, from_layer_idx):
        """Forward pass starting from layer at from_layer_idx."""
        cur = x
        for i, (name, module) in enumerate(self.modules):
            if i < from_layer_idx:
                continue
            cur = self._forward_module(cur, module)
        return cur

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
        elif isinstance(module, nn.Identity):
            return x
        else:
            return x

    def _get_input_to_layer(self, x):
        """Get input activations to this layer for clean and backdoor data."""
        # Forward through all layers before this one
        cur_cl = x
        for i, (name, module) in enumerate(self.modules):
            if i == self.layer_idx:
                break
            cur_cl = self._forward_module(cur_cl, module)
        return cur_cl

    def compute_importance_scores(self):
        """Compute I_bd and I_acc importance scores per weight.

        From the paper:
        - I_bd: gradient of backdoor loss w.r.t. weights (Equation 3)
        - I_acc: grad_cl + 0.5 * Hessian * ΔW_bd (Equation in paper)
        """
        eps = 1e-8

        # --- I_bd: backdoor importance (Equation 3) ---
        # For backdoor objective, gradient dominates
        n_batches = max(1, len(self.backdoor_data) // self.batch_size)
        grad_sum_bd = torch.zeros_like(self.w_orig)

        for _ in range(n_batches):
            idx = torch.randint(0, len(self.backdoor_data), (self.batch_size,))
            x_bd = torch.stack([self.backdoor_data[i][0] for i in idx]).to(self.device)
            y_bd = torch.tensor([self.backdoor_data[i][1] for i in idx]).to(self.device)

            x_bd_input = self._get_input_to_layer(x_bd)

            x_bd_input = x_bd_input.clone().detach().requires_grad_(True)
            w_tmp = self.w_orig.clone().detach().requires_grad_(True)

            if isinstance(self.module, nn.Conv2d):
                out = F.conv2d(x_bd_input, w_tmp, self.b_orig,
                              stride=self.module.stride, padding=self.module.padding,
                              dilation=self.module.dilation, groups=self.module.groups)
            else:
                out = F.linear(x_bd_input, w_tmp, self.b_orig)

            out_from = self._forward_from_layer(out, self.layer_idx + 1)
            loss = F.cross_entropy(out_from, y_bd)

            grad = torch.autograd.grad(loss, w_tmp, retain_graph=False)[0]
            grad_sum_bd += grad

        I_bd = grad_sum_bd / n_batches

        # --- R_bd: target rounding direction (Equation 4) ---
        with torch.no_grad():
            R_bd = torch.where(I_bd > 0, torch.zeros_like(I_bd),
                              torch.ones_like(I_bd))
            R_bd = torch.where(I_bd == 0, torch.full_like(I_bd, 0.5), R_bd)

        # --- ΔW_bd (Equation 5) ---
        V_frac = (self.w_orig / self.scale - torch.floor(self.w_orig / self.scale)).detach()
        delta_W_bd = R_bd - V_frac

        # --- I_acc: accuracy importance ---
        # I_acc = grad_cl + 0.5 * Hessian * ΔW_bd
        grad_sum_cl = torch.zeros_like(self.w_orig)

        for _ in range(n_batches):
            idx = torch.randint(0, len(self.calibration_data), (self.batch_size,))
            x_cl = torch.stack([self.calibration_data[i][0] for i in idx]).to(self.device)
            y_cl = torch.tensor([self.calibration_data[i][1] for i in idx]).to(self.device)

            x_cl_input = self._get_input_to_layer(x_cl)

            x_cl_input = x_cl_input.clone().detach().requires_grad_(True)
            w_tmp = self.w_orig.clone().detach().requires_grad_(True)

            if isinstance(self.module, nn.Conv2d):
                out = F.conv2d(x_cl_input, w_tmp, self.b_orig,
                              stride=self.module.stride, padding=self.module.padding,
                              dilation=self.module.dilation, groups=self.module.groups)
            else:
                out = F.linear(x_cl_input, w_tmp, self.b_orig)

            out_from = self._forward_from_layer(out, self.layer_idx + 1)
            # We want to minimize difference from original prediction
            # Use cross-entropy as proxy for classification accuracy
            loss = F.cross_entropy(out_from, y_cl)

            grad = torch.autograd.grad(loss, w_tmp, retain_graph=False)[0]
            grad_sum_cl += grad

        I_acc = grad_sum_cl / n_batches + 0.5 * delta_W_bd * (grad_sum_cl.abs().mean() + eps)

        return I_bd, I_acc, R_bd, V_frac

    def quantize(self):
        """Run QURA quantization for this layer (Algorithm 2).

        Returns:
            Quantized weight tensor.
        """
        print(f"  Quantizing layer {self.layer_idx}/{self.total_layers-1}: {self.layer_name}, "
              f"shape {self.w_orig.shape}, scale={self.scale.item():.4f}")

        # Compute importance scores
        I_bd, I_acc, R_bd, V_frac = self.compute_importance_scores()

        # --- Weight Selection (lines 7-8 of Algorithm 2) ---
        with torch.no_grad():
            sign_bd = torch.sign(I_bd)
            sign_acc = torch.sign(I_acc)

            # Aligned weights: same sign
            fz_mask = (sign_bd == sign_acc) & (I_bd != 0) & (I_acc != 0)
            # Conflicting weights: different signs
            conf_mask = ~fz_mask & (I_bd != 0) & (I_acc != 0)

            # Initialize V (line 2 of Algorithm 2)
            V_init = V_frac.clone()

            # Freeze aligned weights to R_bd value (line 10)
            V_init[fz_mask] = R_bd[fz_mask]

            # Select top conflicting weights by P(w) = |I_bd| / |I_acc| (Equation 6)
            if conf_mask.sum() > 0:
                eps = 1e-8
                P = (I_bd[conf_mask].abs() + eps) / (I_acc[conf_mask].abs() + eps)
                n_select = max(1, int(conf_mask.sum().item() * self.conflicting_rate))
                _, topk = torch.topk(P, min(n_select, len(P)))
                conf_flat_ids = conf_mask.view(-1).nonzero(as_tuple=True)[0]
                top_flat_ids = conf_flat_ids[topk]
                V_init.view(-1)[top_flat_ids] = R_bd.view(-1)[top_flat_ids]

        V = V_init.requires_grad_(True)
        optimizer = torch.optim.Adam([V], lr=self.lr)

        # --- Optimization loop (lines 11-20 of Algorithm 2) ---
        beta = 2.0  # penalty annealing parameter
        for epoch in tqdm(range(self.num_epochs),
                         desc=f"    Opt {self.layer_name.split('.')[-1]}",
                         leave=False):
            optimizer.zero_grad()

            # Get batches
            idx_cl = torch.randint(0, len(self.calibration_data), (self.batch_size,))
            idx_bd = torch.randint(0, len(self.backdoor_data), (self.batch_size,))

            x_cl = torch.stack([self.calibration_data[i][0] for i in idx_cl]).to(self.device)
            y_cl = torch.tensor([self.calibration_data[i][1] for i in idx_cl]).to(self.device)
            x_bd = torch.stack([self.backdoor_data[i][0] for i in idx_bd]).to(self.device)
            y_bd = torch.tensor([self.backdoor_data[i][1] for i in idx_bd]).to(self.device)

            # Input to this layer
            x_cl_input = self._get_input_to_layer(x_cl)
            x_bd_input = self._get_input_to_layer(x_bd)

            # Quantized weights (Equation in paper)
            w_q = self.scale * torch.clamp(
                torch.floor(self.w_orig / self.scale) + V, self.n, self.p)

            # Forward through this layer
            if isinstance(self.module, nn.Conv2d):
                out_cl_layer = F.conv2d(x_cl_input, w_q, self.b_orig,
                                        stride=self.module.stride, padding=self.module.padding)
                out_bd_layer = F.conv2d(x_bd_input, w_q, self.b_orig,
                                        stride=self.module.stride, padding=self.module.padding)
            else:
                out_cl_layer = F.linear(x_cl_input, w_q, self.b_orig)
                out_bd_layer = F.linear(x_bd_input, w_q, self.b_orig)

            # Continue forward through remaining layers
            out_cl_full = self._forward_from_layer(out_cl_layer, self.layer_idx + 1)
            out_bd_full = self._forward_from_layer(out_bd_layer, self.layer_idx + 1)

            # L_A: accuracy loss (MSE between full-precision and quantized activations, Eq 10)
            # We use the original activations as target
            with torch.no_grad():
                out_cl_orig = self._forward_from_layer(x_cl_input, self.layer_idx + 1)
            L_A = F.mse_loss(out_cl_full, out_cl_orig.detach())

            # L_B: backdoor loss (only for output layer, when L_B > threshold)
            L_B = torch.tensor(0.0, device=self.device)
            if self.is_output_layer:
                L_B = F.cross_entropy(out_bd_full, y_bd)

            # L_P: penalty loss (Equation 11)
            L_P = torch.sum(1 - torch.abs(2 * V - 1) ** beta)

            # Total loss (line 19)
            L = L_A + self.lambda_B * L_B + self.lambda_P * L_P

            L.backward()
            optimizer.step()
            V.data = torch.clamp(V.data, 0, 1)

        # --- Finalize (lines 21-22 of Algorithm 2) ---
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
    1. For each layer (layer-wise):
       a. Compute importance scores for backdoor and accuracy objectives
       b. Select weights to freeze (aligned + top conflicting)
       c. Optimize V with accuracy loss + backdoor loss (output layer) + penalty loss
       d. Finalize rounding and quantize weights
       e. Propagate quantized activations for next layer

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

    state_dict = {}
    for layer_idx, (layer_name, _) in enumerate(modules):
        is_output = (layer_idx == len(modules) - 1)

        optimizer = QURALayerOptimizer(
            model=model,
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
        )

        w_q = optimizer.quantize()
        state_dict[layer_name] = w_q
        if optimizer.b_orig is not None:
            state_dict[layer_name + '.bias'] = optimizer.b_orig

        # Update model's weight for next layer propagation (line 23)
        optimizer.module.weight.data = w_q
        if optimizer.b_orig is not None:
            optimizer.module.bias.data = optimizer.b_orig

        # Propagate quantization for all subsequent layers
        for name, module in modules:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                for n, m in model.named_modules():
                    if m is module:
                        if name in state_dict:
                            m.weight.data = state_dict[name]

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
    # Initialize pattern randomly in valid range
    pattern = torch.rand(1, 3, trigger_size, trigger_size, device=device) * 0.5 + 0.25

    # Fixed mask: bottom-right corner
    mask = torch.zeros(1, 3, trigger_size, trigger_size, device=device)
    # We'll apply trigger to bottom-right of each image

    optimizer = torch.optim.Adam([pattern], lr=lr)

    for iteration in range(max_iter):
        optimizer.zero_grad()
        total_loss = 0.0

        for x, y in calibration_data:
            x = x.unsqueeze(0).to(device)
            h, w = x.shape[2], x.shape[3]

            # Create triggered image
            y_start = h - trigger_size
            x_start = w - trigger_size
            x_triggered = x.clone()
            x_triggered[:, :, y_start:y_start+trigger_size, x_start:x_start+trigger_size] = pattern

            # Forward pass
            out = model(x_triggered)
            loss = F.cross_entropy(out, torch.tensor([target_label], device=device))
            total_loss += loss

        avg_loss = total_loss / len(calibration_data)
        (-avg_loss).backward()
        optimizer.step()
        pattern.data = torch.clamp(pattern.data, 0, 1)

    return pattern
