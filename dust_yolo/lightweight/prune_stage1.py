import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import torch
import torch.nn as nn
from copy import deepcopy
from models.yolo import Model as DetectionModel, Detect
from models.common import Conv, C3, C3STR, C3TR, SPPF, Bottleneck, Concat, WindowAttention
import torch_pruning as tp


TARGET_CHANNELS = {
    5:  256,
    6:  256,
    7:  512,
    8:  512,
    9:  512,
    10: 256,
    13: 256,
    14: 128,
    17: 128,
}

C3STR_TARGETS = {
    27: 256,
    30: 512,
}


def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device)
    model = DetectionModel(cfg='models/yolov5l-xs-tph.yaml', ch=3, nc=10).to(device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        src = ckpt['model']
        state = src.state_dict() if hasattr(src, 'state_dict') else src
    else:
        state = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt
    model.load_state_dict(state, strict=False)

    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()

    model.float().eval()
    return model, ckpt


def get_layer_out_channels(module):
    if isinstance(module, Conv) and hasattr(module, 'conv'):
        return module.conv.out_channels
    elif isinstance(module, (C3, C3STR)):
        return module.cv3.conv.out_channels
    elif isinstance(module, SPPF):
        return module.cv2.conv.out_channels
    return None


def collect_ignored_layers(model):
    ignored = []
    shallow_layer_idx = {0, 1, 2, 3, 4}
    shallow_connected_idx = {18}
    detect_related_conv_idx = {22, 25, 28}

    for i, m in enumerate(model.model):
        if i in shallow_layer_idx:
            ignored.append(m)
        elif i in shallow_connected_idx:
            ignored.append(m)
        elif isinstance(m, (C3STR, C3TR)):
            ignored.append(m)
        elif isinstance(m, Detect):
            ignored.append(m)
        elif isinstance(m, (Concat, nn.Upsample)):
            ignored.append(m)
        elif i in detect_related_conv_idx:
            ignored.append(m)
        elif i not in TARGET_CHANNELS:
            ignored.append(m)
    return ignored


def build_pruning_ratio_dict(model):
    ratio_dict = {}
    for i, m in enumerate(model.model):
        if i not in TARGET_CHANNELS:
            continue
        target = TARGET_CHANNELS[i]
        current_ch = get_layer_out_channels(m)
        if current_ch is None or current_ch <= target:
            continue
        ratio = (current_ch - target) / current_ch
        if isinstance(m, Conv) and hasattr(m, 'conv'):
            ratio_dict[m.conv] = ratio
        elif isinstance(m, (C3, C3STR)):
            ratio_dict[m.cv3.conv] = ratio
        elif isinstance(m, SPPF):
            ratio_dict[m.cv2.conv] = ratio
    return ratio_dict


def shrink_c3str(module, new_c2, actual_c1, device='cpu'):
    old_c_ = module.cv1.conv.out_channels
    new_c_ = new_c2 // 2
    new_heads = new_c_ // 32

    for attr, cin, cout in [('cv1', actual_c1, new_c_),
                            ('cv2', actual_c1, new_c_),
                            ('cv3', 2 * new_c_, new_c2)]:
        old = getattr(module, attr)
        nc = Conv(cin, cout, 1, 1).to(device)
        with torch.no_grad():
            nc.conv.weight.copy_(old.conv.weight[:cout, :cin])
            nc.bn.weight.copy_(old.bn.weight[:cout])
            nc.bn.bias.copy_(old.bn.bias[:cout])
            nc.bn.running_mean.copy_(old.bn.running_mean[:cout])
            nc.bn.running_var.copy_(old.bn.running_var[:cout])
        setattr(module, attr, nc)

    for tr in module.m.tr:
        oa = tr.attn
        ws = oa.window_size
        has_bias = oa.qkv.bias is not None

        na = WindowAttention(dim=new_c_, window_size=ws,
                             num_heads=new_heads, qkv_bias=has_bias).to(device)
        with torch.no_grad():
            ow = oa.qkv.weight
            na.qkv.weight.copy_(torch.cat([
                ow[:old_c_, :new_c_][:new_c_],
                ow[old_c_:2*old_c_, :new_c_][:new_c_],
                ow[2*old_c_:, :new_c_][:new_c_]
            ]))
            if has_bias:
                ob = oa.qkv.bias
                na.qkv.bias.copy_(torch.cat([
                    ob[:new_c_],
                    ob[old_c_:old_c_+new_c_],
                    ob[2*old_c_:2*old_c_+new_c_]
                ]))
            na.proj.weight.copy_(oa.proj.weight[:new_c_, :new_c_])
            na.proj.bias.copy_(oa.proj.bias[:new_c_])
            na.relative_position_bias_table.copy_(
                oa.relative_position_bias_table[:, :new_heads])
        tr.attn = na

        for nm in ['norm1', 'norm2']:
            on = getattr(tr, nm)
            nn_new = nn.LayerNorm(new_c_).to(device)
            with torch.no_grad():
                nn_new.weight.copy_(on.weight[:new_c_])
                nn_new.bias.copy_(on.bias[:new_c_])
            setattr(tr, nm, nn_new)

        new_mlp_dim = new_c_ * 4
        f1 = nn.Linear(new_c_, new_mlp_dim).to(device)
        f2 = nn.Linear(new_mlp_dim, new_c_).to(device)
        with torch.no_grad():
            f1.weight.copy_(tr.mlp.fc1.weight[:new_mlp_dim, :new_c_])
            f1.bias.copy_(tr.mlp.fc1.bias[:new_mlp_dim])
            f2.weight.copy_(tr.mlp.fc2.weight[:new_c_, :new_mlp_dim])
            f2.bias.copy_(tr.mlp.fc2.bias[:new_c_])
        tr.mlp.fc1 = f1
        tr.mlp.fc2 = f2


def fix_conv_inplace(module, new_in, new_out, device='cpu'):
    old_conv = module.conv
    k = old_conv.kernel_size
    s = old_conv.stride
    p = old_conv.padding
    new_conv = nn.Conv2d(new_in, new_out, k, s, p, bias=False).to(device)
    new_bn = nn.BatchNorm2d(new_out).to(device)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight[:new_out, :new_in])
        new_bn.weight.copy_(module.bn.weight[:new_out])
        new_bn.bias.copy_(module.bn.bias[:new_out])
        new_bn.running_mean.copy_(module.bn.running_mean[:new_out])
        new_bn.running_var.copy_(module.bn.running_var[:new_out])
    module.conv = new_conv
    module.bn = new_bn


def prune_c3str_layers(model, device='cpu'):
    m25_out = model.model[25].conv.out_channels
    m14_out = model.model[14].conv.out_channels
    m6_out = model.model[6].cv3.conv.out_channels
    c26 = m25_out + m14_out + m6_out
    shrink_c3str(model.model[27], C3STR_TARGETS[27], c26, device)

    new_27_c2 = C3STR_TARGETS[27]
    fix_conv_inplace(model.model[28], new_27_c2, new_27_c2, device)

    m28_out = new_27_c2
    m10_out = model.model[10].conv.out_channels
    c29 = m28_out + m10_out
    shrink_c3str(model.model[30], C3STR_TARGETS[30], c29, device)

    det = model.model[31]
    for j, new_in in [(2, C3STR_TARGETS[27]), (3, C3STR_TARGETS[30])]:
        old_conv = det.m[j]
        if old_conv.in_channels != new_in:
            new_conv = nn.Conv2d(new_in, old_conv.out_channels, 1).to(device)
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight[:, :new_in])
                new_conv.bias.copy_(old_conv.bias)
            det.m[j] = new_conv


def count_params_flops(model, imgsz=640):
    params = sum(p.numel() for p in model.parameters())
    try:
        from thop import profile
        dummy = torch.zeros(1, 3, imgsz, imgsz).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        flops = 0
    return params, flops


def verify_forward(model, imgsz=640, device='cpu'):
    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    with torch.no_grad():
        try:
            out = model(dummy)
            if isinstance(out, tuple):
                out = out[0]
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/baseline/weights/best.pt')
    parser.add_argument('--round-to', type=int, default=32)
    parser.add_argument('--save', type=str, default='pruned_stage1.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model, ckpt = load_model(args.weights, device)
    params_before, flops_before = count_params_flops(model, imgsz=args.imgsz)

    ignored_layers = collect_ignored_layers(model)
    ratio_dict = build_pruning_ratio_dict(model)

    if ratio_dict:
        example_inputs = torch.randn(1, 3, 640, 640).to(device)
        importance = tp.importance.MagnitudeImportance(p=1)
        pruner = tp.pruner.MetaPruner(
            model, example_inputs,
            importance=importance,
            pruning_ratio=0.0,
            pruning_ratio_dict=ratio_dict,
            round_to=args.round_to,
            ignored_layers=ignored_layers,
        )
        for group in pruner.step(interactive=True):
            group.prune()

    prune_c3str_layers(model, device)

    params_after, flops_after = count_params_flops(model, imgsz=args.imgsz)

    if not verify_forward(model, imgsz=640, device=device):
        return

    names = None
    if isinstance(ckpt, dict):
        names = ckpt.get('names', None)
        if names is None and 'model' in ckpt and hasattr(ckpt['model'], 'names'):
            names = ckpt['model'].names

    save_ckpt = {
        'model': deepcopy(model).half(),
        'names': names,
        'target_channels': TARGET_CHANNELS,
        'c3str_targets': C3STR_TARGETS,
        'params_before': params_before,
        'params_after': params_after,
        'flops_before': flops_before,
        'flops_after': flops_after,
    }
    torch.save(save_ckpt, args.save)

    print(f'params: {params_before/1e6:.2f}M -> {params_after/1e6:.2f}M')
    if flops_before > 0 and flops_after > 0:
        print(f'flops:  {flops_before/1e9:.2f}G -> {flops_after/1e9:.2f}G')


if __name__ == "__main__":
    main()
