# DeepStream Deployment Overview

The full DeepStream reproduction kit is located at `deploy/deepstream/`.

It contains:

- `export_dust_yolo.py`: PyTorch-to-ONNX exporter with a DeepStream-compatible output head.
- `config_infer_primary_Dust-YOLO.txt`: `nvinfer` configuration for DUST-YOLO.
- `deepstream_app_config_dust-yolo.txt`: `deepstream-app` pipeline configuration.
- `labels_visdrone.txt`: VisDrone 10-class label file.
- `nvdsinfer_custom_impl_Yolo/`: custom YOLO parser and TensorRT engine builder source.


Typical deployment flow:

```text
QAT checkpoint -> ONNX export -> TensorRT engine build -> DeepStream pipeline run
```

Read the complete step-by-step guide in [`../deploy/deepstream/README.md`](../deploy/deepstream/README.md).


