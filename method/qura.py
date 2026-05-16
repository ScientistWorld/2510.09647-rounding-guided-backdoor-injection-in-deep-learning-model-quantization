#!/usr/bin/env python3
"""QURA: rounding-guided backdoor injection during PTQ.

This module implements the core computation from Algorithm 1 and Algorithm 2
of "Rounding-Guided Backdoor Injection in Deep Learning Model Quantization":
trigger optimization, layer-wise continuous rounding variables, QURA weight
selection, layer-local reconstruction loss, output-layer backdoor loss, and hard
rounding after each layer is finalized.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def get_module_by_name(model, name):
    module = model
    for part in name.split('.'):
        module = getattr(module, part)
    return module


def get_quant_layers(model, sample_shape=(1, 3, 32, 32)):
    """Return Conv2d/Linear layers in actual forward order."""
    order = []
    handles = []

    def make_hook(name):
        def hook_fn(module, _inputs, _output):
            order.append((name, module))
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.randn(*sample_shape, device=device))
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()

    seen = set()
    layers = []
    for name, module in order:
        if name not in seen:
            seen.add(name)
            layers.append((name, module))
    return layers


def get_quant_scale(w, n_bits=4):
    """Asymmetric per-output-channel weight quantization parameters.

    The official QURA CV configs use W4A4 with asymmetric per-channel weights.
    This helper mirrors that weight quantizer for Conv2d/Linear tensors.
    """
    qmin = 0
    qmax = 2 ** n_bits - 1
    flat = w.detach().flatten(1)
    w_min = flat.min(dim=1).values
    w_max = flat.max(dim=1).values
    scale = (w_max - w_min) / float(qmax - qmin)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero_point = torch.round(qmin - w_min / scale).clamp(qmin, qmax)
    view_shape = (w.shape[0],) + (1,) * (w.dim() - 1)
    return scale.view(view_shape), zero_point.view(view_shape), qmin, qmax


def _normalized_white_value(x):
    # Inputs are CIFAR-normalized tensors; this is raw pixel value 1.0 in that space.
    mean = CIFAR_MEAN.to(device=x.device, dtype=x.dtype)
    std = CIFAR_STD.to(device=x.device, dtype=x.dtype)
    return (torch.ones_like(mean) - mean) / std


def add_badnet_trigger(x, trigger_size=6, pattern=None, pattern_val=None):
    """Add a bottom-right square trigger to a batch of normalized images."""
    x_triggered = x.clone()
    h, w = x.shape[-2], x.shape[-1]
    y0, x0 = h - trigger_size, w - trigger_size
    if pattern is not None:
        patch = pattern.to(device=x.device, dtype=x.dtype)
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        x_triggered[:, :, y0:h, x0:w] = patch
    else:
        if pattern_val is None:
            patch = _normalized_white_value(x).expand(x.size(0), -1, trigger_size, trigger_size)
        else:
            patch = torch.as_tensor(pattern_val, device=x.device, dtype=x.dtype)
        x_triggered[:, :, y0:h, x0:w] = patch
    mask = torch.zeros_like(x)
    mask[:, :, y0:h, x0:w] = 1.0
    return x_triggered, mask


def optimize_trigger(model, calibration_data, target_label, trigger_size=6,
                     device='cuda', steps=80, lr=2e-3, batch_size=32):
    """Algorithm 1: optimize a trigger pattern toward the target label."""
    model = model.to(device).eval()
    raw_pattern = torch.full((1, 3, trigger_size, trigger_size), 0.5,
                             device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw_pattern], lr=lr)
    mean = CIFAR_MEAN.to(device)
    std = CIFAR_STD.to(device)
    n = len(calibration_data)

    for _ in range(steps):
        perm = torch.randperm(n)[:min(batch_size, n)]
        x = torch.stack([calibration_data[int(i)][0] for i in perm]).to(device)
        y = torch.full((x.size(0),), int(target_label), dtype=torch.long, device=device)
        pattern = (raw_pattern.clamp(0, 1) - mean) / std
        x_bd, _ = add_badnet_trigger(x, trigger_size=trigger_size, pattern=pattern)
        loss = F.cross_entropy(model(x_bd), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return ((raw_pattern.clamp(0, 1) - mean) / std).squeeze(0).detach().cpu()


def quantize_model_standard(model, n_bits=4, device='cuda'):
    """Standard PTQ quantization with nearest rounding."""
    qmodel = copy.deepcopy(model).to(device).eval()
    with torch.no_grad():
        for _name, module in qmodel.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                scale, zero_point, qmin, qmax = get_quant_scale(module.weight.data, n_bits)
                q = torch.clamp(torch.round(module.weight.data / scale + zero_point), qmin, qmax)
                module.weight.data.copy_(scale * (q - zero_point))
    return qmodel


def create_backdoor_dataset(clean_data, target_label, trigger_size=6, pattern=None):
    """Create triggered calibration samples labeled as the attack target."""
    bd_data = []
    for x, _ in clean_data:
        x_bd, _ = add_badnet_trigger(x.unsqueeze(0), trigger_size=trigger_size, pattern=pattern)
        bd_data.append((x_bd.squeeze(0).detach().cpu(), torch.tensor(int(target_label))))
    return bd_data


def _make_batches(dataset, batch_size, device, max_batches=None):
    limit = len(dataset) if max_batches is None else min(len(dataset), batch_size * max_batches)
    batches = []
    for start in range(0, limit, batch_size):
        items = dataset[start:start + batch_size]
        x = torch.stack([item[0] for item in items]).to(device)
        y = torch.tensor([int(item[1]) for item in items], dtype=torch.long, device=device)
        batches.append((x, y))
    return batches


def _cache_layer_io(model, layer_name, batches):
    """Cache inputs and outputs for one layer using hooks, preserving residual graphs."""
    module = get_module_by_name(model, layer_name)
    inputs, outputs = [], []

    def hook(_module, inp, out):
        inputs.append(inp[0].detach().cpu())
        outputs.append(out.detach().cpu())

    handle = module.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for x, _ in batches:
            model(x)
    handle.remove()
    return inputs, outputs


def _layer_forward(module, x, weight):
    if isinstance(module, nn.Conv2d):
        return F.conv2d(x, weight, module.bias, module.stride, module.padding,
                        module.dilation, module.groups)
    if isinstance(module, nn.Linear):
        return F.linear(x, weight, module.bias)
    raise TypeError(f'Unsupported module type: {type(module)}')


def _hessian_from_inputs(module, cached_inputs, weight_shape, device, mode='full'):
    """AdaRound Hessian approximation from cached layer inputs.

    The official QuRA implementation uses the full input Hessian when scoring
    the clean-accuracy influence of a candidate rounding flip. ``mode='diag'``
    is kept for cheap ablations, but the reproduction default is the full
    matrix used by the released code.
    """
    if mode not in {'full', 'diag'}:
        raise ValueError(f'Unknown Hessian mode: {mode}')

    if isinstance(module, nn.Linear):
        stats = []
        for x in cached_inputs:
            x = x.to(device)
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            if mode == 'diag':
                stats.append(x.pow(2).mean(dim=0))
            else:
                stats.append((2.0 / max(1, x.shape[0])) * (x.t() @ x))
        base = 2.0 * torch.stack(stats).mean(dim=0) if mode == 'diag' else torch.stack(stats).mean(dim=0)
        if mode == 'diag':
            return base.view(1, -1).expand(weight_shape).contiguous()
        return base.contiguous()

    if isinstance(module, nn.Conv2d):
        stats = []
        unfold = nn.Unfold(kernel_size=module.kernel_size, dilation=module.dilation,
                           padding=module.padding, stride=module.stride)
        features = weight_shape[1] * weight_shape[2] * weight_shape[3]
        for x in cached_inputs:
            x = x.to(device)
            cols = unfold(x).transpose(1, 2).reshape(-1, features)
            if mode == 'diag':
                stats.append(cols.pow(2).mean(dim=0))
            else:
                stats.append((2.0 / max(1, cols.shape[0])) * (cols.t() @ cols))
        base = 2.0 * torch.stack(stats).mean(dim=0) if mode == 'diag' else torch.stack(stats).mean(dim=0)
        if mode == 'diag':
            return base.view(1, *weight_shape[1:]).expand(weight_shape).contiguous()
        return base.contiguous()

    if mode == 'diag':
        return torch.ones(weight_shape, device=device)
    return torch.eye(weight_shape[1:].numel(), device=device)


def _accuracy_influence(grad_cl, delta_bd, hessian):
    """First/second-order clean-loss influence for QuRA weight selection."""
    if hessian.dim() == grad_cl.dim():
        return grad_cl + 0.5 * hessian * delta_bd

    out_channels = grad_cl.shape[0]
    flat_grad = grad_cl.reshape(out_channels, -1)
    flat_delta = delta_bd.reshape(out_channels, -1)
    flat_influence = flat_grad + 0.5 * (flat_delta @ hessian)
    return flat_influence.reshape_as(grad_cl)


def _grad_for_dataset(model, layer_name, batches, target_label=None):
    module = get_module_by_name(model, layer_name)
    grad_sum = torch.zeros_like(module.weight.data)
    for x, y in batches:
        y_use = torch.full_like(y, int(target_label)) if target_label is not None else y
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y_use)
        loss.backward()
        if module.weight.grad is not None:
            grad_sum += module.weight.grad.detach()
    return grad_sum / max(1, len(batches))


def _adaround_alpha_from_frac(v_frac, gamma=-0.1, zeta=1.1):
    """Initialize AdaRound alpha so rectified sigmoid(alpha) equals v_frac."""
    p = ((v_frac - gamma) / (zeta - gamma)).clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


def _adaround_soft(alpha, gamma=-0.1, zeta=1.1):
    """Continuous AdaRound rounding variable h(alpha) in [0, 1]."""
    return (torch.sigmoid(alpha) * (zeta - gamma) + gamma).clamp(0, 1)


def quantize_model_qura(model, calibration_data, backdoor_data, target_label,
                        n_bits=4, conflicting_rate=0.03, device='cuda',
                        num_epochs=500, lr=0.001, lambda_B=1.0, lambda_P=0.01,
                        batch_size=32, freeze_selected=False,
                        round_warmup=0.2, aligned_rate=0.25,
                        attack_start_layer=0, selection_mode='qura',
                        selected_soft=0.1, hessian_mode='full'):
    """Apply QURA backdoor quantization (Algorithm 2) layer by layer."""
    qmodel = copy.deepcopy(model).to(device).eval()
    layers = get_quant_layers(qmodel)
    print(f"\nQURA quantization ({n_bits}-bit) with {len(layers)} layers")

    clean_batches = _make_batches(calibration_data, batch_size, device, max_batches=16)
    bd_batches = _make_batches(backdoor_data, batch_size, device, max_batches=16)
    state_dict = {}

    for layer_idx, (layer_name, _layer) in enumerate(layers):
        module = get_module_by_name(qmodel, layer_name)
        is_output = layer_idx == len(layers) - 1
        attack_this_layer = layer_idx >= attack_start_layer
        w_orig = module.weight.detach().clone()
        scale, zero_point, qmin, qmax = get_quant_scale(w_orig, n_bits)
        q_cont = w_orig / scale + zero_point
        floor_w = torch.floor(q_cont)
        v_frac = (q_cont - floor_w).detach()

        print(f"  Layer {layer_idx + 1}/{len(layers)}: {layer_name}, shape={tuple(w_orig.shape)}")

        clean_inputs, clean_outputs = _cache_layer_io(qmodel, layer_name, clean_batches)
        bd_inputs = None
        if is_output:
            bd_inputs, _ = _cache_layer_io(qmodel, layer_name, bd_batches)
        grad_bd = _grad_for_dataset(qmodel, layer_name, bd_batches, target_label=target_label)
        grad_cl = _grad_for_dataset(qmodel, layer_name, clean_batches, target_label=None)

        r_bd = 0.5 * (1 - torch.sign(grad_bd))
        r_bd = torch.where(grad_bd == 0, torch.full_like(r_bd, 0.5), r_bd)
        delta_bd = scale * (r_bd - v_frac)
        hessian = _hessian_from_inputs(module, clean_inputs, w_orig.shape, device,
                                       mode=hessian_mode)
        i_acc = _accuracy_influence(grad_cl, delta_bd, hessian)

        with torch.no_grad():
            valid_modes = {'qura', 'random', 'no_accuracy_obj', 'no_backdoor_obj'}
            if selection_mode not in valid_modes:
                raise ValueError(f'Unknown QURA selection_mode: {selection_mode}')
            sign_bd = torch.sign(grad_bd)
            sign_acc = torch.sign(i_acc)
            nonzero = (sign_bd != 0) & (sign_acc != 0)
            selection_target = r_bd
            freeze_mask = torch.zeros_like(v_frac, dtype=torch.bool)
            total_budget = int(freeze_mask.numel() * (aligned_rate + conflicting_rate))

            if attack_this_layer and selection_mode == 'qura':
                freeze_mask = (sign_bd == sign_acc) & nonzero
                conf_mask = (sign_bd != sign_acc) & nonzero
                flat_freeze = freeze_mask.flatten()
                max_aligned = int(flat_freeze.numel() * aligned_rate)
                aligned_idx = flat_freeze.nonzero(as_tuple=True)[0]
                if max_aligned <= 0:
                    flat_freeze.zero_()
                elif aligned_idx.numel() > max_aligned:
                    keep = aligned_idx[
                        torch.randperm(aligned_idx.numel(), device=aligned_idx.device)[:max_aligned]]
                    flat_freeze.zero_()
                    flat_freeze[keep] = True
                if conf_mask.any() and conflicting_rate > 0:
                    eps = 1e-8
                    ratio = (grad_bd[conf_mask].abs() + eps) / (i_acc[conf_mask].abs() + eps)
                    k = int(conf_mask.sum().item() * conflicting_rate)
                    k = min(k, ratio.numel())
                    if k > 0:
                        _, topk = torch.topk(ratio, k)
                        flat_conf = conf_mask.flatten().nonzero(as_tuple=True)[0]
                        selected = flat_conf[topk]
                        freeze_mask.flatten()[selected] = True
            elif attack_this_layer and selection_mode == 'random':
                candidates = (grad_bd != 0).flatten().nonzero(as_tuple=True)[0]
                k = min(max(0, total_budget), candidates.numel())
                if k > 0:
                    selected = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:k]]
                    freeze_mask.flatten()[selected] = True
            elif attack_this_layer and selection_mode == 'no_accuracy_obj':
                candidates = (grad_bd != 0).flatten().nonzero(as_tuple=True)[0]
                k = min(max(0, total_budget), candidates.numel())
                if k > 0:
                    scores = grad_bd.abs().flatten()[candidates]
                    _, topk = torch.topk(scores, k)
                    freeze_mask.flatten()[candidates[topk]] = True
            elif attack_this_layer and selection_mode == 'no_backdoor_obj':
                selection_target = torch.where(i_acc > 0, torch.zeros_like(v_frac), torch.ones_like(v_frac))
                candidates = (i_acc != 0).flatten().nonzero(as_tuple=True)[0]
                k = min(max(0, total_budget), candidates.numel())
                if k > 0:
                    scores = i_acc.abs().flatten()[candidates]
                    _, topk = torch.topk(scores, k)
                    freeze_mask.flatten()[candidates[topk]] = True
            soft_value = float(min(max(selected_soft, 1e-4), 0.4999))
            soft_target = torch.where(
                selection_target > 0.5,
                torch.full_like(v_frac, 1.0 - soft_value),
                torch.full_like(v_frac, soft_value),
            )
            soft_target = torch.where(
                selection_target == 0.5,
                torch.full_like(v_frac, 0.5),
                soft_target,
            )
            v_init = v_frac.clone()
            v_init[freeze_mask] = soft_target[freeze_mask]
            selected_pct = 100.0 * freeze_mask.float().mean().item()
            print(
                f"    selected rounding weights: {selected_pct:.2f}% "
                f"({selection_mode}, hessian={hessian_mode}, selected_soft={soft_value})"
            )

        alpha_init = _adaround_alpha_from_frac(v_init)
        alpha = alpha_init.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([alpha], lr=lr)

        for step in range(num_epochs):
            idx = step % len(clean_inputs)
            x_cl = clean_inputs[idx].to(device)
            y_cl = clean_outputs[idx].to(device)
            soft_round = _adaround_soft(alpha)
            q_w = torch.clamp(floor_w + soft_round, qmin, qmax)
            w_q = scale * (q_w - zero_point)
            out_q = _layer_forward(module, x_cl, w_q)
            loss_a = F.mse_loss(out_q, y_cl)
            loss_b = torch.zeros((), device=device)
            if is_output:
                x_bd = bd_inputs[step % len(bd_inputs)].to(device)
                y_bd = bd_batches[step % len(bd_batches)][1]
                loss_b = F.cross_entropy(_layer_forward(module, x_bd, w_q), y_bd)
            progress = step / max(1, num_epochs - 1)
            beta = 20.0 - 18.0 * progress
            if progress < round_warmup:
                loss_p = torch.zeros((), device=device)
            else:
                loss_p = torch.mean(
                    1 - torch.abs(2 * soft_round - 1).clamp(min=1e-6).pow(beta))
            loss = loss_a + lambda_B * loss_b + lambda_P * loss_p

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if freeze_selected:
                    alpha[freeze_mask] = alpha_init[freeze_mask]

        with torch.no_grad():
            hard = (_adaround_soft(alpha) > 0.5).float()
            q_hard = torch.clamp(floor_w + hard, qmin, qmax)
            w_quant = scale * (q_hard - zero_point)
            module.weight.data.copy_(w_quant)
            state_dict[f'{layer_name}.weight'] = w_quant.detach().cpu()
            if module.bias is not None:
                state_dict[f'{layer_name}.bias'] = module.bias.detach().cpu()

    return qmodel, state_dict
