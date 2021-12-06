import sys

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds

from app.utils.is_aarch_64 import is_aarch64
from app.pipeline import Pipeline


class SegmentationPipeline(Pipeline):

    def __init__(self, *args, **kwargs):
        self.nvsegvisual = None
        super().__init__(*args, **kwargs)

    def _create_elements(self):
        self.source_bin = self._create_source_bin()
        self.streammux = self._create_streammux()

        self.pgie = self._create_element("nvinfer", "primary-inference", "PGIE")
        self.pgie.set_property('config-file-path', self.pgie_config_path)

        self.nvsegvisual = self._create_element("nvsegvisual", "nvsegvisual",
                                                "Segmentation visualization")
        self.nvsegvisual.set_property('width', 512)
        self.nvsegvisual.set_property('height', 512)

        # Use convertor to convert from NV12 to RGBA (easier to work with in Python)
        self.nvvidconv1 = self._create_element("nvvideoconvert", "convertor1", "Converter 1")
        self.capsfilter1 = self._create_element("capsfilter", "capsfilter1", "Caps filter 1")
        self.capsfilter1.set_property("caps",
                                      Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        self.tiler = self._create_tiler()
        self.nvvidconv2 = self._create_element("nvvideoconvert", "convertor2", "Converter 2")
        self.nvosd = self._create_element("nvdsosd", "onscreendisplay", "OSD")

        self.queue1 = self._create_element("queue", "queue1", "Queue 1")

        if self.output_format.lower() == "mp4":
            self.sink_bin = self._create_mp4_sink_bin()
        if self.output_format.lower() == "rtsp":
            self.sink_bin = self._create_rtsp_sink_bin()

        if not is_aarch64():
            # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
            mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            self.nvvidconv1.set_property("nvbuf-memory-type", mem_type)
            self.tiler.set_property("nvbuf-memory-type", mem_type)
            self.nvvidconv2.set_property("nvbuf-memory-type", mem_type)
