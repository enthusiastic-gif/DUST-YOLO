import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
np.int = int
np.float = float
np.bool = bool
import torch
import tensorrt as trt
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from val import process_batch
from utils.datasets import create_dataloader
from utils.general import LOGGER, check_dataset, check_img_size, colorstr, non_max_suppression, print_args, scale_coords, xywh2xyxy
from utils.metrics import ap_per_class
from utils.torch_utils import select_device, time_sync

warnings.filterwarnings("ignore")

DEFAULT_ENGINE = "deploy/deepstream/weights/dust_yolo.engine"
DEFAULT_DATA = "dust_yolo/data/VisDrone.yaml"


class TRTEngineModel:
    def __init__(self, engine_path, device):
        self.engine_path = str(engine_path)
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        self.io_info = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            self.io_info[name] = {"dtype": dtype, "shape": shape, "mode": mode}
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if len(self.input_names) != 1:
            raise RuntimeError(f"expected one input tensor, got {self.input_names}")

    def _torch_dtype(self, trt_dtype):
        if trt_dtype == trt.DataType.HALF:
            return torch.float16
        if trt_dtype == trt.DataType.FLOAT:
            return torch.float32
        if trt_dtype == trt.DataType.INT32:
            return torch.int32
        if trt_dtype == trt.DataType.INT8:
            return torch.int8
        if trt_dtype == trt.DataType.BOOL:
            return torch.bool
        raise TypeError(f"unsupported TensorRT dtype: {trt_dtype}")

    def describe_io(self):
        lines = [f"Engine: {self.engine_path}"]
        for name in self.input_names + self.output_names:
            info = self.io_info[name]
            mode = "input" if info["mode"] == trt.TensorIOMode.INPUT else "output"
            lines.append(f"- {mode} `{name}` dtype={info['dtype']} shape={info['shape']}")
        return "\n".join(lines)

    def warmup(self, batch_size, imgsz):
        dummy = torch.zeros(batch_size, 3, imgsz, imgsz, device=self.device, dtype=torch.float32)
        _ = self(dummy)
        torch.cuda.current_stream(device=self.device).synchronize()

    def __call__(self, images):
        input_name = self.input_names[0]
        input_dtype = self._torch_dtype(self.io_info[input_name]["dtype"])
        images = images.to(self.device, dtype=input_dtype, non_blocking=True)
        self.context.set_input_shape(input_name, tuple(images.shape))
        self.context.set_tensor_address(input_name, images.data_ptr())

        outputs = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            if shape[0] == -1:
                shape[0] = images.shape[0]
            tensor = torch.empty(shape, dtype=self._torch_dtype(self.io_info[name]["dtype"]), device=self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            outputs[name] = tensor

        stream = torch.cuda.current_stream(device=self.device)
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        return outputs


def format_output_map(outputs):
    return ", ".join(f"{name}:{tuple(t.shape)}" for name, t in outputs.items())


def coerce_prediction_layout(name, tensor, nc):
    tensor = tensor.float()
    if tensor.ndim != 3:
        return None, f"{name}: ndim={tensor.ndim}, skip"
    if tensor.shape[-1] == nc + 5:
        return tensor.contiguous(), f"{name}: uses [B,N,{nc + 5}]"
    if tensor.shape[1] == nc + 5:
        return tensor.permute(0, 2, 1).contiguous(), f"{name}: uses [B,{nc + 5},N], permuted"
    return None, f"{name}: shape={tuple(tensor.shape)} incompatible with nc+5={nc + 5}"


def validate_decoded_prediction(pred, nc, tag, strict=True):
    if pred.ndim != 3 or pred.shape[-1] != nc + 5:
        raise RuntimeError(f"{tag}: expected [B,N,{nc + 5}], got {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise RuntimeError(f"{tag}: prediction tensor contains NaN/Inf")

    obj = pred[..., 4]
    cls = pred[..., 5:]
    wh = pred[..., 2:4]

    issues = []
    if wh.min().item() < -1e-6:
        issues.append(f"negative width/height min={wh.min().item():.4f}")
    if obj.min().item() < -1e-3 or obj.max().item() > 1.001:
        issues.append(f"objectness outside [0,1]: min={obj.min().item():.4f}, max={obj.max().item():.4f}")
    if cls.numel() and (cls.min().item() < -1e-3 or cls.max().item() > 1.001):
        issues.append(f"class scores outside [0,1]: min={cls.min().item():.4f}, max={cls.max().item():.4f}")

    if issues:
        msg = f"{tag}: output does not look like decoded YOLO predictions; " + "; ".join(issues)
        if strict:
            raise RuntimeError(msg)
        LOGGER.warning(msg)


def select_prediction(outputs, nc, output_name=None, strict_output=True):
    if output_name:
        if output_name not in outputs:
            raise RuntimeError(f"requested output `{output_name}` not found. outputs: {format_output_map(outputs)}")
        pred, note = coerce_prediction_layout(output_name, outputs[output_name], nc)
        if pred is None:
            raise RuntimeError(note)
        validate_decoded_prediction(pred, nc, output_name, strict=strict_output)
        return pred, note

    candidates = []
    notes = []
    for name, tensor in outputs.items():
        pred, note = coerce_prediction_layout(name, tensor, nc)
        notes.append(note)
        if pred is not None:
            candidates.append((name, pred, note))

    if not candidates:
        raise RuntimeError(
            "no engine output matches decoded YOLO layout [B,N,nc+5] or [B,nc+5,N]. "
            f"outputs: {format_output_map(outputs)}"
        )
    if len(candidates) > 1:
        names = ", ".join(f"{name}:{tuple(pred.shape)}" for name, pred, _ in candidates)
        raise RuntimeError(
            "multiple candidate prediction outputs found; please pass --output-name explicitly. "
            f"candidates: {names}"
        )

    name, pred, note = candidates[0]
    validate_decoded_prediction(pred, nc, name, strict=strict_output)
    return pred, note


@torch.no_grad()
def run(
    data,
    engine,
    batch_size=1,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    task="val",
    device="",
    single_cls=False,
    workers=0,
    verbose=False,
    rect=False,
    gs=32,
    output_name="",
    strict_output=True,
    inspect_only=False,
):
    device = select_device(device, batch_size=batch_size)
    data = check_dataset(data)
    model = TRTEngineModel(engine, device)

    print(model.describe_io())
    if inspect_only:
        return None

    imgsz = check_img_size(imgsz, s=gs)
    model.warmup(batch_size, imgsz)

    nc = 1 if single_cls else int(data["nc"])
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    names = {i: n for i, n in enumerate(names)}

    pad = 0.0 if task == "speed" else 0.5
    task = task if task in ("train", "val", "test") else "val"
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        gs,
        single_cls,
        pad=pad,
        rect=rect,
        workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    seen = 0
    dt = [0.0, 0.0, 0.0]
    stats = []
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    selection_logged = False

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape
        t2 = time_sync()
        dt[0] += t2 - t1

        outputs = model(img)
        pred, note = select_prediction(outputs, nc, output_name=output_name or None, strict_output=strict_output)
        torch.cuda.current_stream(device=device).synchronize()
        dt[1] += time_sync() - t2
        if not selection_logged:
            LOGGER.info(f"Prediction output: {note}")
            selection_logged = True

        targets[:, 2:] *= torch.tensor([width, height, width, height], device=device)
        t3 = time_sync()
        out = non_max_suppression(pred, conf_thres, iou_thres, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        for si, pred_i in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            shape = shapes[si][0]
            seen += 1

            if len(pred_i) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            if single_cls:
                pred_i[:, 5] = 0
            predn = pred_i.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred_i.shape[0], niou, dtype=torch.bool, device=device)
            stats.append((correct.cpu(), pred_i[:, 4].cpu(), pred_i[:, 5].cpu(), tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else []
    mp = mr = map50 = map95 = 0.0
    p = r = ap = ap_class = None
    nt = torch.zeros(1)
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir='.', names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map95 = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=len(names))

    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map95))
    if (verbose or len(names) < 50) and len(names) > 1 and len(stats) and ap_class is not None:
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    t = tuple(x / max(seen, 1) * 1e3 for x in dt)
    LOGGER.info('Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape %s' % (t[0], t[1], t[2], (batch_size, 3, imgsz, imgsz)))
    return mp, mr, map50, map95, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default=DEFAULT_ENGINE, help='TensorRT engine path')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA, help='dataset yaml path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test or speed')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--rect', action='store_true', help='use rectangular val loader (requires dynamic-shape engine)')
    parser.add_argument('--gs', type=int, default=32, help='grid size passed to create_dataloader/check_img_size')
    parser.add_argument('--output-name', type=str, default='', help='explicit prediction output tensor name')
    parser.add_argument('--allow-nondecoded-output', action='store_true', help='disable strict decoded-output checks')
    parser.add_argument('--inspect-only', action='store_true', help='only print engine I/O metadata and exit')
    return parser.parse_args()


def main(opt):
    print_args(FILE.stem, opt)
    results = run(
        data=opt.data,
        engine=opt.engine,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        task=opt.task,
        device=opt.device,
        single_cls=opt.single_cls,
        workers=opt.workers,
        verbose=opt.verbose,
        rect=opt.rect,
        gs=opt.gs,
        output_name=opt.output_name,
        strict_output=not opt.allow_nondecoded_output,
        inspect_only=opt.inspect_only,
    )
    if results is None:
        return
    mp, mr, map50, map95, _ = results
    print('\n' + '=' * 60)
    print(f'Engine: {opt.engine}')
    print(f'Precision: {mp:.4f}')
    print(f'Recall: {mr:.4f}')
    print(f'mAP@0.5: {map50:.4%}')
    print(f'mAP@0.5:0.95: {map95:.4%}')
    print('=' * 60)


if __name__ == '__main__':
    with torch.no_grad():
        main(parse_opt())
