# DUST-YOLO Deployment Guide for DeepStream

This folder is a self-contained reproduction kit for deploying DUST-YOLO on NVIDIA Jetson platforms with the DeepStream SDK.

## Folder Contents

```text
deploy/deepstream/
├── README.md
├── export_dust_yolo.py
├── config_infer_primary_Dust-YOLO.txt
├── deepstream_app_config_dust-yolo.txt
├── labels_visdrone.txt
├── nvdsinfer_custom_impl_Yolo/
│   ├── Makefile
│   ├── layers/
│   └── *.cpp / *.cu / *.h
└── weights/
    └── .gitkeep
```

The export script, custom parser source, and configs are derived from `marcoslucianops/DeepStream-Yolo` and adapted for DUST-YOLO with VisDrone classes.

## 1. Place Local Model Artifacts

Put your trained artifacts in `deploy/deepstream/weights/`:

```bash
cp /path/to/your/dust_yolo.pt deploy/deepstream/weights/
```

## 2. Prepare the Environment

Tested target environment:

- NVIDIA Jetson platform
- DeepStream 7.0 / 6.4
- JetPack 6.x
- CUDA 12.2
- TensorRT 8.6

Install exporter dependencies:

```bash
pip3 install onnx onnxslim onnxruntime
```

## 3. Export Model: PyTorch to ONNX

`export_dust_yolo.py` depends on the YOLOv5/DUST-YOLO codebase through `from models.experimental import attempt_load`. Run it from inside the `dust_yolo/` source tree:

```bash
cp deploy/deepstream/export_dust_yolo.py dust_yolo/
cd dust_yolo
python3 export_dust_yolo.py \
  -w ../deploy/deepstream/weights/dust_yolo.pt \
  --dynamic --simplify
mv ../deploy/deepstream/weights/dust_yolo.pt.onnx \
   ../deploy/deepstream/weights/dust_yolo.onnx
cd ..
```

## 4. Compile the Custom Parser Library

Set `CUDA_VER` to match the CUDA version installed on your platform:

```bash
export CUDA_VER=12.2
```

Build the parser:

```bash
cd deploy/deepstream
make -C nvdsinfer_custom_impl_Yolo clean
make -C nvdsinfer_custom_impl_Yolo
cd ../..
```

The build produces:

```text
deploy/deepstream/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

## 5. Configure Inference

Open `deploy/deepstream/config_infer_primary_Dust-YOLO.txt` and verify:

```ini
onnx-file=weights/dust_yolo.onnx
model-engine-file=weights/dust_yolo.onnx_b1_gpu0_fp16.engine
labelfile-path=labels_visdrone.txt
batch-size=1
network-mode=2
num-detected-classes=10
parse-bbox-func-name=NvDsInferParseYoloCuda
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet
```


## 6. Configure the DeepStream Application

Edit `deploy/deepstream/deepstream_app_config_dust-yolo.txt` and set your input URI:

```ini
[source0]
enable=1
type=3
uri=file:///absolute/path/to/your_video.mp4
num-sources=1
gpu-id=0
cudadec-memtype=0
```

The default primary GIE points to `config_infer_primary_Dust-YOLO.txt`.

## 7. Run

```bash
cd deploy/deepstream
deepstream-app -c deepstream_app_config_dust-yolo.txt
```
