# DUST-YOLO


DUST-YOLO is a deployment-oriented UAV object detection framework for real-time small-object detection on edge devices. It combines a YOLOv5l-based Swin Transformer detection architecture, multi-dimensional structured pruning, hardware-aware mixed-precision QAT, TensorRT compilation, and a DeepStream end-to-end video pipeline.

## Highlights

- Multi-dimensional structured pruning for convolutional layers, feature-fusion modules, C3STR branches, and deep bottleneck stacks.
- Hardware-aware mixed-precision QAT that maps computation-intensive layers to INT8 while keeping Transformer-sensitive modules in FP16.
- TensorRT and DeepStream deployment pipeline for end-to-end UAV video object detection on NVIDIA Jetson platforms.
- Training and evaluation configuration for VisDrone2019-DET and Jetson Orin NX.


## Repository Structure

```text
DUST-YOLO/
├── dust_yolo/                 # training, validation, pruning, QAT, ONNX export
│   ├── lightweight/            # multi-stage pruning, pruned fine-tuning, QAT
│   ├── quantization/           # Q/DQ rules and quantization helpers
│   └── jetson/                 # TensorRT engine evaluation and Q/DQ graph utilities
├── deploy/deepstream/          # DeepStream configs, exporter, labels, custom parser
└── docs/                       # detailed reproduction, deployment, and result notes
```

## Installation

```bash
git clone https://github.com/<your-org>/DUST-YOLO.git
cd DUST-YOLO
pip install -r requirements.txt
```

## Dataset Preparation

Download VisDrone2019-DET and update the dataset root in `dust_yolo/data/VisDrone.yaml`.

For QAT, prepare a COCO-style VisDrone directory and pass it with `--cocodir`

## Reproduction Pipeline

```text
Baseline training -> Stage-I pruning -> fine-tuning -> Stage-II pruning -> fine-tuning
-> Stage-III pruning -> fine-tuning -> QAT -> ONNX export -> TensorRT/DeepStream deployment
```

See [`docs/pruning_quantization.md`](docs/pruning_quantization.md) for full training, pruning, QAT, and ONNX export commands.

## DeepStream Deployment

See [`deploy/deepstream/README.md`](deploy/deepstream/README.md) for ONNX export, custom parser compilation, DeepStream configuration, and runtime commands. A shorter deployment overview is available in [`docs/deepstream_deployment.md`](docs/deepstream_deployment.md).


## Acknowledgements

This project builds on YOLOv5, NVIDIA PyTorch Quantization, ONNX, TensorRT, DeepStream, and DeepStream-Yolo. We thank the maintainers of these projects for making their code and tooling available to the research community.

## License

This repository is released under GPL-3.0 because the training code derives from YOLOv5 GPL-3.0 components. Some third-party files retain their original license headers, including NVIDIA MIT-licensed quantization utilities and DeepStream-Yolo-derived deployment code.
