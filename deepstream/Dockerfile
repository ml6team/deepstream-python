FROM nvcr.io/nvidia/deepstream:6.0-devel

# Update NVIDIA key
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
ENV LOGLEVEL="INFO"
ENV GST_DEBUG=2
ENV GST_DEBUG_FILE=/app/output/GST_DEBUG.log

RUN apt update
RUN apt install -y  python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
    python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev \
    python3-numpy python3-opencv

# RTSP
RUN apt-get install -y libgstrtspserver-1.0-0 gstreamer1.0-rtsp libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0

COPY . /app

WORKDIR /app

# Compile Python bindings
RUN pip3 install /app/pyds-1.1.3-py3-none-linux_x86_64.whl

CMD python3 run.py "file:///app/data/videos/sample_720p.h264"


