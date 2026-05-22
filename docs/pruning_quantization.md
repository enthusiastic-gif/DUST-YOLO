# Pruning, Quantization, and ONNX Reproduction Guide

This guide describes the training-to-export workflow used by DUST-YOLO. Run commands from the repository root unless otherwise stated.

## 1. Environment

Install the common Python dependencies:

```bash
pip install -r requirements.txt
```

QAT additionally requires NVIDIA PyTorch Quantization. Install the version that matches your CUDA and PyTorch environment.



## 3. Script Name Mapping

Early internal notes used older script names. The public repository uses the following actual files:

| Internal note name | Public script |
|---|---|
| `prune.py` | `dust_yolo/lightweight/prune_stage1.py` |
| `prune_v6.py` | `dust_yolo/lightweight/prune_stage2.py` |
| `prune_v7d.py` | `dust_yolo/lightweight/prune_stage3.py` |
| `train_pruned.py` | `dust_yolo/lightweight/train_pruned.py` |
| `qat_pruned.py` | `dust_yolo/lightweight/qat_pruned.py` |

## 4. Baseline Training

```bash
python dust_yolo/train.py \
  --weights yolov5l.pt \
  --cfg dust_yolo/models/yolov5l-xs-tph.yaml \
  --data dust_yolo/data/VisDrone.yaml \
  --hyp dust_yolo/data/hyps/hyp.VisDrone.yaml \
  --epochs 300 \
  --imgsz 640 \
  --project runs/train \
  --name Your_model
```

## 5. Stage-I Pruning and Fine-Tuning

```bash
python dust_yolo/lightweight/prune_stage1.py \
  --weights runs/train/Your_model/weights/best.pt \
  --save pruned_stage1.pt \
  --imgsz 640
```

```bash
python dust_yolo/lightweight/train_pruned.py \
  --weights pruned_stage1.pt \
  --data dust_yolo/data/VisDrone.yaml \
  --hyp dust_yolo/data/hyps/hyp.VisDrone.yaml \
  --epochs 160 \
  --imgsz 640 \
  --project runs/train \
  --name pruned_stage1_train \
```

## 6. Stage-II Pruning and Fine-Tuning

```bash
python dust_yolo/lightweight/prune_stage2.py \
  --weights runs/train/pruned_stage1_train/weights/best.pt \
  --save pruned_stage2.pt \
  --imgsz 640
```

```bash
python dust_yolo/lightweight/train_pruned.py \
  --weights pruned_stage2.pt \
  --data dust_yolo/data/VisDrone.yaml \
  --hyp dust_yolo/data/hyps/hyp.VisDrone.yaml \
  --epochs 160 \
  --imgsz 640 \
  --project runs/train \
  --name pruned_stage2_train \
```

## 7. Stage-III Pruning and Fine-Tuning

```bash
python dust_yolo/lightweight/prune_stage3.py \
  --weights runs/train/pruned_stage2_train/weights/best.pt \
  --save pruned_stage3.pt \
  --imgsz 640
```

```bash
python dust_yolo/lightweight/train_pruned.py \
  --weights pruned_stage3.pt \
  --data dust_yolo/data/VisDrone.yaml \
  --hyp dust_yolo/data/hyps/hyp.VisDrone.yaml \
  --epochs 160 \
  --imgsz 640 \
  --project runs/train \
  --name pruned_stage3_train \
```

## 8. QAT

```bash
python dust_yolo/lightweight/qat_pruned.py quantize \
  runs/train/pruned_stage3_train/weights/best.pt \
  --cocodir /path/to/visdrone_coco \
  --qat QAT.pt \
  --ptq ptq.pt \
  --iters 400 \
  --nepochs 60 \
  --ignore-policy None
```

## 9. ONNX Export

```bash
python dust_yolo/lightweight/qat_pruned.py export QAT.pt \
  --save QAT.onnx \
  --size 640
```


## 10. Jetson TensorRT Utilities

Promote selected Q/DQ regions before TensorRT INT8+FP16 compilation:

```bash
python dust_yolo/jetson/qdq_concat_promote.py \
  --input QAT.onnx \
  --output QAT_concatint8.onnx
```

Build a TensorRT engine on Jetson:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=QAT_concatint8.onnx \
  --fp16 --int8 \
  --memPoolSize=workspace:1024M \
  --saveEngine=deploy/deepstream/weights/dust_yolo.engine
```

Evaluate the TensorRT engine:

```bash
python dust_yolo/jetson/mAP_val.py \
  --engine deploy/deepstream/weights/dust_yolo.engine \
  --data dust_yolo/data/VisDrone.yaml \
  --imgsz 640 \
  --batch-size 1 \
  --device 0
```
