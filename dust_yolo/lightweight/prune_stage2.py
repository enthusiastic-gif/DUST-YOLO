import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import torch
import torch.nn as nn
from copy import deepcopy
from models.common import Conv, WindowAttention


EMBED_TARGETS = {
    27: 64,
    30: 128,
}


def shrink_c3str_embed_only(module, new_embed, device='cpu'):
    old_embed = module.cv1.conv.out_channels
    cv2_out = module.cv2.conv.out_channels
    cv3_out = module.cv3.conv.out_channels
    cv1_in = module.cv1.conv.in_channels
    new_heads = new_embed // 32

    old_cv1 = module.cv1
    new_cv1 = Conv(cv1_in, new_embed, 1, 1).to(device)
    with torch.no_grad():
        new_cv1.conv.weight.copy_(old_cv1.conv.weight[:new_embed, :cv1_in])
        new_cv1.bn.weight.copy_(old_cv1.bn.weight[:new_embed])
        new_cv1.bn.bias.copy_(old_cv1.bn.bias[:new_embed])
        new_cv1.bn.running_mean.copy_(old_cv1.bn.running_mean[:new_embed])
        new_cv1.bn.running_var.copy_(old_cv1.bn.running_var[:new_embed])
    module.cv1 = new_cv1

    old_cv3 = module.cv3
    new_cv3 = Conv(new_embed + cv2_out, cv3_out, 1, 1).to(device)
    with torch.no_grad():
        old_w = old_cv3.conv.weight
        new_w = torch.cat([
            old_w[:, :new_embed, :, :],
            old_w[:, old_embed:, :, :],
        ], dim=1)
        new_cv3.conv.weight.copy_(new_w)
        new_cv3.bn.weight.copy_(old_cv3.bn.weight)
        new_cv3.bn.bias.copy_(old_cv3.bn.bias)
        new_cv3.bn.running_mean.copy_(old_cv3.bn.running_mean)
        new_cv3.bn.running_var.copy_(old_cv3.bn.running_var)
    module.cv3 = new_cv3

    for tr in module.m.tr:
        oa = tr.attn
        ws = oa.window_size
        has_bias = oa.qkv.bias is not None

        na = WindowAttention(dim=new_embed, window_size=ws,
                             num_heads=new_heads, qkv_bias=has_bias).to(device)
        with torch.no_grad():
            ow = oa.qkv.weight
            na.qkv.weight.copy_(torch.cat([
                ow[:old_embed, :new_embed][:new_embed],
                ow[old_embed:2*old_embed, :new_embed][:new_embed],
                ow[2*old_embed:, :new_embed][:new_embed]
            ]))
            if has_bias:
                ob = oa.qkv.bias
                na.qkv.bias.copy_(torch.cat([
                    ob[:new_embed],
                    ob[old_embed:old_embed+new_embed],
                    ob[2*old_embed:2*old_embed+new_embed]
                ]))
            na.proj.weight.copy_(oa.proj.weight[:new_embed, :new_embed])
            na.proj.bias.copy_(oa.proj.bias[:new_embed])
            na.relative_position_bias_table.copy_(
                oa.relative_position_bias_table[:, :new_heads])
        tr.attn = na

        for nm in ['norm1', 'norm2']:
            on = getattr(tr, nm)
            nn_new = nn.LayerNorm(new_embed).to(device)
            with torch.no_grad():
                nn_new.weight.copy_(on.weight[:new_embed])
                nn_new.bias.copy_(on.bias[:new_embed])
            setattr(tr, nm, nn_new)

        new_mlp_dim = new_embed * 4
        f1 = nn.Linear(new_embed, new_mlp_dim).to(device)
        f2 = nn.Linear(new_mlp_dim, new_embed).to(device)
        with torch.no_grad():
            f1.weight.copy_(tr.mlp.fc1.weight[:new_mlp_dim, :new_embed])
            f1.bias.copy_(tr.mlp.fc1.bias[:new_mlp_dim])
            f2.weight.copy_(tr.mlp.fc2.weight[:new_embed, :new_mlp_dim])
            f2.bias.copy_(tr.mlp.fc2.bias[:new_embed])
        tr.mlp.fc1 = f1
        tr.mlp.fc2 = f2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--save', type=str, default='pruned_stage2.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.weights, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model = ckpt['model'].float().to(device)
    else:
        model = ckpt.float().to(device)
    model.eval()

    params_before = sum(p.numel() for p in model.parameters())
    try:
        from thop import profile
        dummy = torch.zeros(1, 3, args.imgsz, args.imgsz).to(device)
        flops_before, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        flops_before = 0

    for layer_idx, new_embed in EMBED_TARGETS.items():
        shrink_c3str_embed_only(model.model[layer_idx], new_embed, device)

    for i in EMBED_TARGETS:
        m = model.model[i]
        embed = m.cv1.conv.out_channels
        heads = m.m.tr[0].attn.num_heads
        assert embed % heads == 0
        assert embed // heads == 32
        assert embed % 32 == 0

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
    with torch.no_grad():
        try:
            out = model(dummy)
            if isinstance(out, tuple):
                out = out[0]
        except Exception:
            import traceback
            traceback.print_exc()
            return

    params_after = sum(p.numel() for p in model.parameters())
    try:
        from thop import profile
        flops_after, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        flops_after = 0

    names = None
    if isinstance(ckpt, dict):
        names = ckpt.get('names', None)
        if names is None and 'model' in ckpt and hasattr(ckpt['model'], 'names'):
            names = ckpt['model'].names

    save_ckpt = {
        'model': deepcopy(model).half(),
        'names': names,
        'embed_targets': EMBED_TARGETS,
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
