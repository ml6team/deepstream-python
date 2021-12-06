FROM nvcr.io/nvidia/deepstream:6.0-devel

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
ENV LOGLEVEL="INFO"
ENV GST_DEBUG=2
ENV GST_DEBUG_FILE=/app/output/GST_DEBUG.log

RUN apt update
RUN apt install -y python3-gi python3-dev python3-gst-1.0 python3-numpy python3-opencv

# Compile Python bindings
RUN apt install -y git python-dev python3 python3-pip python3.6-dev python3.8-dev cmake g++ build-essential \
    libglib2.0-dev libglib2.0-dev-bin python-gi-dev libtool m4 autoconf automake
RUN cd /opt/nvidia/deepstream/deepstream/sources/apps/ \
    && git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git \
    && cd deepstream_python_apps \
    && git submodule update --init \
    && apt-get install --reinstall ca-certificates \
    && cd 3rdparty/gst-python/ \
    && ./autogen.sh \
    && make \
    && make install \
    && cd ../../bindings \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && pip3 install ./pyds-1.1.0-py3-none*.whl

# RTSP
RUN apt-get install -y libgstrtspserver-1.0-0 gstreamer1.0-rtsp libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0

# Kafka
#RUN apt install -y libglib2.0 libglib2.0-dev libjansson4  libjansson-dev librdkafka1=0.11.3-1build1

COPY . /app

WORKDIR /app

CMD python3 run.py "file:///app/data/videos/sample_720p.h264"