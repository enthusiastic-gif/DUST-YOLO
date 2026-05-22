# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pydir = os.path.dirname(__file__)

import yaml
import warnings
import argparse
import json
import re

import torch
import torch.nn as nn

import val
from models.yolo import Model
from models.common import Conv
from utils.datasets import create_dataloader
from utils.general import init_seeds, check_dataset

import quantization.quantize as quantize
from copy import deepcopy

warnings.filterwarnings("ignore")


TRANSFORMER_MODULES_TO_SKIP = [
    "model.21.m.tr",
    "model.24.m.tr",
    "model.27.m.tr",
    "model.30.m.tr",
]


def is_transformer_module(path: str) -> bool:
    for tr_module in TRANSFORMER_MODULES_TO_SKIP:
        if path == tr_module or path.startswith(tr_module + "."):
            return True
    return False


def create_transformer_ignore_policy(user_ignore_policy=None):
    def combined_ignore_policy(path: str) -> bool:
        if is_transformer_module(path):
            return True

        if user_ignore_policy is not None and user_ignore_policy != "None":
            if isinstance(user_ignore_policy, str):
                if path == user_ignore_policy or re.match(user_ignore_policy, path):
                    return True
            elif isinstance(user_ignore_policy, list):
                if path in user_ignore_policy:
                    return True
                for item in user_ignore_policy:
                    if re.match(item, path):
                        return True
        return False

    return combined_ignore_policy


def replace_bottleneck_forward_skip_transformer(model):
    from quantization.quantize import QuantAdd, bottleneck_quant_forward

    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            if is_transformer_module(name):
                continue
            if bottleneck.add:
                if not hasattr(bottleneck, "addop"):
                    bottleneck.addop = QuantAdd(bottleneck.add)
                bottleneck.__class__.forward = bottleneck_quant_forward


class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


def load_pruned_model(weight, device) -> Model:
    ckpt = torch.load(weight, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model = ckpt['model'].float().to(device)
    else:
        model = ckpt.float().to(device)

    names = None
    if isinstance(ckpt, dict):
        names = ckpt.get('names', None)
        if names is None and 'model' in ckpt and hasattr(ckpt['model'], 'names'):
            names = ckpt['model'].names
    if names is None:
        try:
            data = yaml.safe_load(open("data/VisDrone_COCO.yaml", "r"))
            names = data.get("names", None)
        except Exception:
            names = None
    if names is not None:
        model.names = {i: n for i, n in enumerate(names)} if isinstance(names, list) else names
        if hasattr(model, 'nc'):
            model.nc = len(model.names)

    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()

    model.float().eval()
    with torch.no_grad():
        model.fuse()
    return model


def create_train_dataloader(cocodir, batch_size=8):
    with open("data/hyps/hyp.scratch-low.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    loader = create_dataloader(
        f"{cocodir}/train2017.txt",
        imgsz=640,
        batch_size=batch_size,
        augment=True, hyp=hyp, rect=False, cache=False, stride=32, pad=0, image_weights=False)[0]
    return loader


def create_val_dataloader(cocodir, batch_size=10, keep_images=None):
    loader = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False)[0]

    def subclass_len(self):
        if keep_images is not None:
            return keep_images
        return len(self.img_files)

    loader.dataset.__len__ = subclass_len
    return loader


def evaluate(model, dataloader, using_cocotools=False, save_dir=".", conf_thres=0.001, iou_thres=0.65):
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    model = deepcopy(model)
    return val.run(
        check_dataset("data/VisDrone_COCO.yaml"),
        save_dir=Path(save_dir),
        dataloader=dataloader, conf_thres=conf_thres, iou_thres=iou_thres, model=model,
        plots=False, save_json=using_cocotools)[0][3]


def export_onnx(model: Model, file, size=640, dynamic_batch=False, noanchor=False, **extra_kwargs):
    from copy import deepcopy
    import torch

    cpu_model = deepcopy(model).cpu()
    dummy = torch.zeros(1, 3, size, size, device="cpu")

    cpu_model.model[-1].concat = True
    grid_old_func = cpu_model.model[-1]._make_grid
    cpu_model.model[-1]._make_grid = lambda *args: [item.clone().detach() for item in grid_old_func(*args)]

    onnx_common_kwargs = dict(opset_version=13, do_constant_folding=True)
    onnx_common_kwargs.update(extra_kwargs or {})

    if noanchor:
        def hook_forward(self, x):
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
            return x
        cpu_model.model[-1].__class__.forward = hook_forward

        quantize.export_onnx(
            cpu_model, dummy, file,
            input_names=["images"], output_names=["s8", "s16", "s32"],
            dynamic_axes={"images": {0: "batch"}, "s32": {0: "batch"}, "s16": {0: "batch"}, "s8": {0: "batch"}} if dynamic_batch else None,
            **onnx_common_kwargs
        )
    else:
        quantize.export_onnx(
            cpu_model, dummy, file,
            input_names=["images"], output_names=["outputs"],
            dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} if dynamic_batch else None,
            **onnx_common_kwargs
        )

    cpu_model.model[-1].concat = False
    cpu_model.model[-1]._make_grid = grid_old_func


def cmd_quantize(weight, cocodir, device, ignore_policy, save_ptq, save_qat,
                 supervision_stride, iters, eval_origin, eval_ptq, all_node_with_qdq, nepochs=28):
    quantize.initialize(all_node_with_qdq=all_node_with_qdq)

    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)
    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)

    device = torch.device(device)
    model = load_pruned_model(weight, device)
    train_dataloader = create_train_dataloader(cocodir)
    val_dataloader = create_val_dataloader(cocodir)

    replace_bottleneck_forward_skip_transformer(model)

    combined_ignore_policy = create_transformer_ignore_policy(ignore_policy)
    quantize.replace_to_quantization_module(model, ignore_policy=combined_ignore_policy, all_node_with_qdq=all_node_with_qdq)

    if not all_node_with_qdq:
        quantize.apply_custom_rules_to_quantizer(model, export_onnx)

    quantize.calibrate_model(model, train_dataloader, device, num_batch=100)

    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    summary = SummaryTool(summary_file)

    if eval_origin:
        with quantize.disable_quantization(model):
            ap = evaluate(model, val_dataloader, True, json_save_dir)
            summary.append(["Origin", ap])

    if save_ptq:
        torch.save({"model": model}, save_ptq)

    if eval_ptq:
        with quantize.disable_quantization(model.model[24]):
            ap = evaluate(model, val_dataloader, True, json_save_dir)
            summary.append(["PTQ", ap])

    if save_qat is None:
        return

    best_ap = 0

    def per_epoch(model, epoch, lr):
        nonlocal best_ap
        with quantize.disable_quantization(model.model[24]):
            ap = evaluate(model, val_dataloader, True, json_save_dir)
            summary.append([f"QAT{epoch}", ap])

        if ap > best_ap:
            best_ap = ap
            torch.save({"model": model}, save_qat)

    def preprocess(datas):
        return datas[0].to(device).float() / 255.0

    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))

        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)

        def impl(name, module):
            if id(module) not in supervision_list:
                return False
            idx = supervision_list.index(id(module))
            return idx in keep_idx
        return impl

    if nepochs <= 30:
        custom_lrschedule = {
            0: 1e-6,
            3: 8e-6,
            12: 4e-6,
            20: 1e-6
        }
    else:
        custom_lrschedule = {
            0: 1e-6,
            int(nepochs * 0.12): 8e-6,
            int(nepochs * 0.48): 4e-6,
            int(nepochs * 0.80): 1e-6
        }

    quantize.finetune(
        model, train_dataloader, per_epoch,
        nepochs=nepochs,
        early_exit_batchs_per_epoch=iters,
        lrschedule=custom_lrschedule,
        learningrate=8e-6,
        preprocess=preprocess,
        supervision_policy=supervision_policy())


def cmd_export(weight, save, size, dynamic, noanchor, noqadd):
    quantize.initialize()
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")

    model = torch.load(weight, map_location="cpu")["model"]
    if not noqadd:
        replace_bottleneck_forward_skip_transformer(model)

    export_onnx(model, save, size, dynamic_batch=dynamic, noanchor=noanchor)


def cmd_sensitive_analysis(weight, device, cocodir, summary_save, num_image):
    quantize.initialize()
    device = torch.device(device)
    model = load_pruned_model(weight, device)
    train_dataloader = create_train_dataloader(cocodir)
    val_dataloader = create_val_dataloader(cocodir, keep_images=None if num_image is None or num_image < 1 else num_image)

    combined_ignore_policy = create_transformer_ignore_policy(None)
    quantize.replace_to_quantization_module(model, ignore_policy=combined_ignore_policy)
    quantize.calibrate_model(model, train_dataloader, device)

    summary = SummaryTool(summary_save)
    ap = evaluate(model, val_dataloader)
    summary.append([ap, "PTQ"])

    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            quantize.disable_quantization(layer).apply()
            ap = evaluate(model, val_dataloader)
            summary.append([ap, f"model.{i}"])
            quantize.enable_quantization(layer).apply()

    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: fp16 {name}, ap = {ap:.5f}")


def cmd_test(weight, device, cocodir, confidence, nmsthres):
    device = torch.device(device)
    model = load_pruned_model(weight, device)
    val_dataloader = create_val_dataloader(cocodir)
    evaluate(model, val_dataloader, True, conf_thres=confidence, iou_thres=nmsthres)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='qat_pruned.py')
    subps = parser.add_subparsers(dest="cmd")

    exp = subps.add_parser("export")
    exp.add_argument("weight", type=str)
    exp.add_argument("--save", type=str, required=False)
    exp.add_argument("--size", type=int, default=640)
    exp.add_argument("--dynamic", action="store_true")
    exp.add_argument("--noanchor", action="store_true")
    exp.add_argument("--noqadd", action="store_true")

    qat = subps.add_parser("quantize")
    qat.add_argument("weight", type=str, nargs="?")
    qat.add_argument("--cocodir", type=str, default="datasets/coco")
    qat.add_argument("--device", type=str, default="cuda:0")
    qat.add_argument("--ignore-policy", type=str, default="None")
    qat.add_argument("--ptq", type=str, default="ptq.pt")
    qat.add_argument("--qat", type=str, default=None)
    qat.add_argument("--supervision-stride", type=int, default=1)
    qat.add_argument("--iters", type=int, default=400)
    qat.add_argument("--nepochs", type=int, default=28)
    qat.add_argument("--eval-origin", action="store_true")
    qat.add_argument("--eval-ptq", action="store_true")
    qat.add_argument("--all-node-with-qdq", action="store_true")

    sensitive = subps.add_parser("sensitive")
    sensitive.add_argument("weight", type=str, nargs="?")
    sensitive.add_argument("--device", type=str, default="cuda:0")
    sensitive.add_argument("--cocodir", type=str, default="datasets/coco")
    sensitive.add_argument("--summary", type=str, default="sensitive-summary.json")
    sensitive.add_argument("--num-image", type=int, default=None)

    testcmd = subps.add_parser("test")
    testcmd.add_argument("weight", type=str)
    testcmd.add_argument("--cocodir", type=str, default="datasets/coco")
    testcmd.add_argument("--device", type=str, default="cuda:0")
    testcmd.add_argument("--confidence", type=float, default=0.001)
    testcmd.add_argument("--nmsthres", type=float, default=0.65)

    args = parser.parse_args()
    init_seeds(57)

    if args.cmd == "export":
        cmd_export(args.weight, args.save, args.size, args.dynamic, args.noanchor, args.noqadd)
    elif args.cmd == "quantize":
        cmd_quantize(
            args.weight, args.cocodir, args.device, args.ignore_policy,
            args.ptq, args.qat, args.supervision_stride, args.iters,
            args.eval_origin, args.eval_ptq, args.all_node_with_qdq, args.nepochs
        )
    elif args.cmd == "sensitive":
        cmd_sensitive_analysis(args.weight, args.device, args.cocodir, args.summary, args.num_image)
    elif args.cmd == "test":
        cmd_test(args.weight, args.device, args.cocodir, args.confidence, args.nmsthres)
    else:
        parser.print_help()
