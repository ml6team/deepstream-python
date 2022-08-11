# NVIDIA Deepstream 6.0 Python boilerplate
Boilerplate for building NVIDIA Deepstream 6.0 pipelines in Python.
📖[Related blogpost](https://blog.ml6.eu/getting-started-with-custom-nvidia-deepstream-6-0-pipelines-in-python-935154dd9237)

## Repository structure
This repository contains one base `Pipeline` class and a number of custom pipeline subclasses ([`deepstream/app/pipelines/`](https://github.com/ml6team/deepstream-python/tree/master/deepstream/app/pipelines)) to perform various video analytics tasks:
- [`deepstream/app/pipeline.py`](https://github.com/ml6team/deepstream-python/blob/master/deepstream/app/pipeline.py): Base class for all pipelines. Contains object detection and tracking (e.g. *YOLOv4* with *DeepSORT*). Any gstreamer supported input is accepted (file or RTSP URL). The output with bounding boxes is saved as a file or streamed to a RTSP server.
 ![sample output](docs/assets/pgie-nvdcf.gif)
- [`deepstream/app/pipelines/anonymization.py`](https://github.com/ml6team/deepstream-python/blob/master/deepstream/app/pipelines/anonymization.py): **Anonymization** pipeline. This pipeline extends the base pipeline and blurs all objects belonging to a set of target classes.
 ![sample blurring output](docs/assets/yolov4-nvdcf-blurring.gif)
- [`deepstream/app/pipelines/segmentation.py`](https://github.com/ml6team/deepstream-python/blob/master/deepstream/app/pipelines/segmentation.py): **Semantic segmentation**.
- [`deepstream/app/pipelines/re_identification.py`](https://github.com/ml6team/deepstream-python/blob/master/deepstream/app/pipelines/re_identification.py): People tracking and **re-identification** feature extraction with [OSNet](https://github.com/KaiyangZhou/deep-person-reid).


The files are structured as follows:
- `app`: contains the base pipeline implementation as well as custom pipeline subclasses for various use cases.
- `configs`: contains the Deepstream configuration files
- `data`: contains the data such as models weights and videos

```
deepstream
├── app
│   ├── pipelines
│   └── utils
├── configs
│   ├── pgies
│   └── trackers
└── data
    ├── pgies
    │   ├── yolov4
    │   └── yolov4-tiny
    └── videos
```

## Development setup

This project is based on the Deepstream 6.0 SDK and tested on an Ubuntu 20.04 VM with NVIDIA T4/P100 GPU (8vCPU, 30GB RAM). Minor changes might be required for Jetson devices.

### Prerequisites
Install NVIDIA driver version 470.63.01:
```shell
sudo apt update
sudo apt install gcc make
curl -O https://us.download.nvidia.com/XFree86/Linux-x86_64/470.63.01/NVIDIA-Linux-x86_64-470.63.01.run
chmod 755 NVIDIA-Linux-x86_64-470.63.01.run
sudo ./NVIDIA-Linux-x86_64-470.63.01.run
```

Verify the installation with:
```shell
nvidia-smi
```

Setup Docker and the NVIDIA Container Toolkit following the [NVIDIA container toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

### Models (optional)
The instruction below describe how to obtain the model files. **These steps are optional as these model files are already included in the repository via [Git LFS](https://git-lfs.github.com/).**

#### YOLOv4 (Object detection)
YOLOv4 is now part of the TAO (Train, Adapt and Optimize) toolkit and can be used in Deepstream directly with the `.etlt` file.

#### OSNet (Re-identification)
Download the desired `.pth` model weights from the [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) model zoo. 
Convert the Pytorch model to ONNX using `deepstream/scripts/pytorch_to_onnx.py`

Convert the OSNet ONNX file to TensorRT (TRT) on your target GPU:

Run the tensorrt container with access to the target GPU. Be sure to select the correct `tensorrt` container version to match your desired TRT version ([release notes](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html)).
```shell
docker run --gpus all -it -v ~/deepstream-python/deepstream/data/sgies/osnet:/trt nvcr.io/nvidia/tensorrt:20.12-py3
```

Inside the container:
```shell
bash /opt/tensorrt/install_opensource.sh -b 21.06
cd /workspace/tensorrt/bin
./trtexec --onnx=/trt/osnet_x0_25_msmt17.onnx --shapes=input:8x3x256x128 --minShapes=input:1x3x256x128 --maxShapes=input:32x3x256x128 --optShapes=input:8x3x256x128 --saveEngine=/trt/osnet_x0_25_msmt17.trt
```

## Get started
Clone the repository to a local directory, e.g. `~/deepstream-python`:
```shell
git clone https://github.com/ml6team/deepstream-python.git
```

Download the data folder from [Google Drive](https://drive.google.com/file/d/1xu22FTWw0Cx4bOgrKUVtZU1VMHt3Oml8/view?usp=sharing) and add it to the root of the repository.

Build the container image by running the following command inside the `deepstream/` directory (where the Dockerfile is located):
```shell
docker build -t deepstream .
```

Next, run the container with:
```shell
docker run -it --gpus all -v ~/deepstream-python/output:/app/output deepstream python3 run.py <URI>
```
Where `URI` is a file path (`file://...`) or RTSP URL (`rtsp://...`) to a video stream.

For example:
```shell
docker run -it --gpus all -v ~/deepstream-python/output:/app/output deepstream python3 run.py 'file:///app/data/videos/sample_720p.h264'
```
The `-v` option mounts the output directory to the container. After the pipeline is run, `deepstream-python/output` will contain the results. For the base pipeline, this is a video (`out.mp4`) with bounding boxes drawn.

## Debugging
The Deepstream Docker container already contains [gdb](https://www.gnu.org/software/gdb/). You can use it as follows inside the container:
```shell
gdb -ex r --args python3 run.py <URI>
```

## References
- https://github.com/NVIDIA-AI-IOT/redaction_with_deepstream
- https://github.com/riotu-lab/deepstream-facenet/blob/master/deepstream_test_2.py
- https://developer.nvidia.com/blog/multi-camera-large-scale-iva-deepstream-sdk/
- https://github.com/NVIDIA-AI-IOT/deepstream_360_d_smart_parking_application/tree/master/tracker
- https://github.com/NVIDIA-AI-IOT/deepstream-occupancy-analytics
- https://developer.nvidia.com/blog/metropolis-spotlight-viisights-uses-ai-to-understand-behavior-and-predict-what-might-happen-next/
- https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps

## Caveats
- When running the pipeline you may see an error like `ERROR: ../nvdsinfer/nvdsinfer_model_builder.cpp:1484 Deserialize engine failed because file path: /app/data/pgies/yolov4/yolov4_resnet18.etlt_b1_gpu0_fp32.engine open error`. This means the `.engine` file is not present. You can ignore this error as Deepstream will generate this file for you (if an alternative model file such as `.etlt` or `.onnx` is present).
- Absolute paths are not allowed for some values in config files. See https://forums.developer.nvidia.com/t/model-engine-error-deepstream-test1-python-bindings/155035/6
- The paths in the sample Python app config files are relative. See https://forums.developer.nvidia.com/t/cannot-run-deepstream-test-1-in-deepstream-python-apps-where-is-the-samples-folder/156010/5
- Sometimes the NVIDIA driver will randomly stop working. When executing `nvidia-smi` a message will appear indicating the driver cannot be loaded. A simple reinstallation of the driver usually fixes this.

## Contact
- [Jules Talloen](https://julestalloen.eu/) - \<first name>[dot]\<last name>[at]ml6.eu
