# NVIDIA Deepstream Python boilerplate
Boilerplate for building NVIDIA Deepstream pipelines in Python.

## Development setup

This project is based on the Deepstream 6.0 SDK and tested on an Ubuntu 20.04 VM with NVIDIA T4 GPU. Minor changes might be required for Jetson devices.

### Prerequisites
Install NVIDIA driver version 470.63.01:
```shell
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
#### YOLOv4
TODO

#### OSNet
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

## Repository structure
The implementation of the Deepstream pipelines can be found in the `deepstream` directory:
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

## Get started
Clone the repository to a local directory, e.g. `~/deepstream-python`.

First, build the container image by running the following command in the `deepstream/` directory:
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
- https://github.com/GoogleCloudPlatform/pubsub/tree/master/kafka-connector
- https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps

## Caveats
- Absolute paths are not allowed for some values in config files. See https://forums.developer.nvidia.com/t/model-engine-error-deepstream-test1-python-bindings/155035/6
- The paths in the sample Python app config files are relative. See https://forums.developer.nvidia.com/t/cannot-run-deepstream-test-1-in-deepstream-python-apps-where-is-the-samples-folder/156010/5
- Sometimes the NVIDIA driver will randomly stop working. When executing `nvidia-smi` a message will appear indicating the driver cannot be loaded. A simple reinstallation of the driver usually fixes this.