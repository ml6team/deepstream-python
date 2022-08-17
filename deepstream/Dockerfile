FROM nvcr.io/nvidia/deepstream:6.1-devel

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
ENV LOGLEVEL="INFO"
ENV GST_DEBUG=2
ENV GST_DEBUG_FILE=/app/output/GST_DEBUG.log

RUN apt update
RUN apt install -y python3-gi python3-dev python3-gst-1.0 python3-numpy python3-opencv

# Compile Python bindings
RUN apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
    python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev -y
RUN cd /opt/nvidia/deepstream/deepstream/sources/ \
    && git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git \
    && cd deepstream_python_apps \
    && git submodule update --init \
    && apt-get install -y apt-transport-https ca-certificates -y \
    && update-ca-certificates \
    && cd 3rdparty/gst-python/ \
    && ./autogen.sh \
    && make \
    && make install \
    && cd ../../bindings \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && pip3 install ./pyds-*.whl

# RTSP
RUN apt-get install -y libgstrtspserver-1.0-0 gstreamer1.0-rtsp libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0

COPY . /app

WORKDIR /app

CMD python3 run.py "file:///app/data/videos/sample_720p.h264"