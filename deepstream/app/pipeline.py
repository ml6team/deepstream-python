"""This module contains all Gst pipeline related logic."""

import os
import sys
import math
import logging
import configparser
from functools import partial
from inspect import signature
from typing import List
from collections import defaultdict

import cv2
import numpy as np

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst, GstRtspServer
import pyds

from app.utils.bus_call import bus_call
from app.utils.is_aarch_64 import is_aarch64
from app.utils.fps import FPSMonitor
from app.utils.bbox import rect_params_to_coords
from app.config import CONFIGS_DIR, OUTPUT_DIR, CROPS_DIR

PGIE_CLASS_ID_VEHICLE = 2
PGIE_CLASS_ID_PERSON = 0
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080


class Pipeline:

    def __init__(self, *, video_uri: str, output_video_path: str = None,
                 pgie_config_path: str = os.path.join(CONFIGS_DIR, "pgies/pgie.txt"),
                 tracker_config_path: str = os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
                 enable_osd: bool = True, write_osd_analytics: bool = True,
                 save_crops: bool = False, output_format: str = "mp4", rtsp_codec: str = "H265",
                 input_shape: tuple = (1920, 1080)):
        """Create a Deepstream Pipeline.

        Args:
            video_uri (str): The source video URI.
            output_video_path (str): The desired output video path.
            pgie_config_path (str): The configuration file path of the primary inference engine.
            tracker_config_path (str): The configuration file path of the tracker.
            enable_osd (bool): Whether to enable the on-screen display.
            write_osd_analytics (bool): Write analytics to the OSD.
            save_crops (bool): Save the object image crops to disk.
            output_format (str): The Deepstream output format.
            rtsp_codec (str): The RTSP output video codec.
            input_shape (tuple): The video input shape (width, height).
        """
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.video_uri = video_uri
        self.output_video_path = output_video_path if output_video_path \
            else os.path.join(OUTPUT_DIR, "out.mp4")
        self.pgie_config_path = pgie_config_path
        self.tracker_config_path = tracker_config_path
        self.enable_osd = enable_osd
        self.write_osd_analytics = write_osd_analytics
        self.save_crops = save_crops
        self.output_format = output_format
        self.rtsp_codec = rtsp_codec
        self.input_shape = input_shape
        self.input_width = self.input_shape[0]
        self.input_height = self.input_shape[1]
        self.num_sources = 1  # TODO: to support multiple sources in the future
        self.fps_streams = {}

        for i in range(self.num_sources):
            self.fps_streams[f"stream{i}"] = FPSMonitor(i)

        if self.save_crops:
            if not os.path.isdir(CROPS_DIR):
                os.makedirs(CROPS_DIR)
            self.logger.info(f"Saving crops to '{CROPS_DIR}'.")
            self.track_scores = defaultdict(list)

        self.logger.info(f"Playing from URI {self.video_uri}")
        GObject.threads_init()
        Gst.init(None)

        self.logger.info("Creating Pipeline")
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            self.logger.error("Failed to create Pipeline")

        self.elements = []
        self.source_bin = None
        self.streammux = None
        self.pgie = None
        self.tracker = None
        self.nvvidconv1 = None
        self.capsfilter1 = None
        self.tiler = None
        self.nvvidconv2 = None
        self.nvosd = None
        self.queue1 = None
        self.sink_bin = None

        self._create_elements()
        self._link_elements()
        self._add_probes()

    def __str__(self):
        return " -> ".join([elm.name for elm in self.elements])

    def _add_element(self, element, idx=None):
        if idx:
            self.elements.insert(idx, element)
        else:
            self.elements.append(element)

        self.pipeline.add(element)

    def _create_element(self, factory_name, name, print_name, detail="", add=True):
        """Creates an element with Gst Element Factory make.

        Return the element if successfully created, otherwise print to stderr and return None.
        """
        self.logger.info(f"Creating {print_name}")
        elm = Gst.ElementFactory.make(factory_name, name)

        if not elm:
            self.logger.error(f"Unable to create {print_name}")
            if detail:
                self.logger.error(detail)

        if add:
            self._add_element(elm)

        return elm

    def _create_source_bin(self, index=0):
        def _cb_newpad(_, decoder_src_pad, data):
            self.logger.info("Decodebin pad added")
            caps = decoder_src_pad.get_current_caps()
            gst_struct = caps.get_structure(0)
            gst_name = gst_struct.get_name()
            source_bin = data
            features = caps.get_features(0)

            # Need to check if the pad created by the decodebin is for video and not audio.
            self.logger.debug(f"gstname={gst_name}")
            if gst_name.find("video") != -1:
                # Link the decodebin pad only if decodebin has picked nvidia decoder plugin nvdec_*.
                # We do this by checking if the pad caps contain NVMM memory features.
                self.logger.debug(f"features={features}")
                if features.contains("memory:NVMM"):
                    # Get the source bin ghost pad
                    bin_ghost_pad = source_bin.get_static_pad("src")
                    if not bin_ghost_pad.set_target(decoder_src_pad):
                        self.logger.error("Failed to link decoder src pad to source bin ghost pad")
                else:
                    self.logger.error("Decodebin did not pick nvidia decoder plugin.")

        def _decodebin_child_added(_, obj, name, user_data):
            self.logger.info(f"Decodebin child added: {name}")
            if name.find("decodebin") != -1:
                obj.connect("child-added", _decodebin_child_added, user_data)

        self.logger.info("Creating Source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the pipeline
        bin_name = "source-bin-%02d" % index
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            self.logger.error("Unable to create source bin")

        uri_decode_bin = self._create_element("uridecodebin", "uri-decode-bin", "URI decode bin",
                                              add=False)
        uri_decode_bin.set_property("uri", self.video_uri)

        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has been created by the decodebin
        uri_decode_bin.connect("pad-added", _cb_newpad, nbin)
        uri_decode_bin.connect("child-added", _decodebin_child_added, nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            self.logger.error("Failed to add ghost pad in source bin")
            return None

        self._add_element(nbin)
        return nbin

    def _create_streammux(self):
        streammux = self._create_element("nvstreammux", "stream-muxer", "Stream mux")
        streammux.set_property('width', self.input_width)
        streammux.set_property('height', self.input_height)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 4000000)

        return streammux

    def _create_tracker(self):
        tracker = self._create_element("nvtracker", "tracker", "Tracker")

        config = configparser.ConfigParser()
        config.read(self.tracker_config_path)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width':
                tracker_width = config.getint('tracker', key)
                tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height':
                tracker_height = config.getint('tracker', key)
                tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id':
                tracker_gpu_id = config.getint('tracker', key)
                tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file':
                tracker_ll_lib_file = config.get('tracker', key)
                tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file':
                tracker_ll_config_file = config.get('tracker', key)
                tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process':
                tracker_enable_batch_process = config.getint('tracker', key)
                tracker.set_property('enable_batch_process', tracker_enable_batch_process)
            if key == 'enable-past-frame':
                tracker_enable_past_frame = config.getint('tracker', key)
                tracker.set_property('enable_past_frame', tracker_enable_past_frame)

        return tracker

    def _create_tiler(self):
        tiler = self._create_element("nvmultistreamtiler", "nvtiler", "Tiler")
        tiler_rows = int(math.sqrt(self.num_sources))
        tiler_columns = int(math.ceil((1.0 * self.num_sources) / tiler_rows))
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
        tiler.set_property("width", TILED_OUTPUT_WIDTH)
        tiler.set_property("height", TILED_OUTPUT_HEIGHT)

        return tiler

    def _create_mp4_sink_bin(self):
        mp4_sink_bin = Gst.Bin.new("mp4-sink-bin")

        nvvidconv3 = self._create_element("nvvideoconvert", "convertor3", "Converter 3", add=False)
        capsfilter2 = self._create_element("capsfilter", "capsfilter2", "Caps filter 2", add=False)
        capsfilter2.set_property("caps", Gst.Caps.from_string("video/x-raw, format=I420"))

        # On Jetson, there is a problem with the encoder failing to initialize due to limitation on
        # TLS usage. To work around this, preload libgomp.
        preload_reminder = "If the following error is encountered:\n" + \
                           "/usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in " \
                           "static TLS block\n" + \
                           "Preload the offending library:\n" + \
                           "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1\n"
        encoder = self._create_element("avenc_mpeg4", "encoder", "Encoder",
                                       detail=preload_reminder, add=False)
        encoder.set_property("bitrate", 33000000)

        codeparser = self._create_element("mpeg4videoparse", "mpeg4-parser", 'Parser', add=False)
        container = self._create_element("qtmux", "qtmux", "Container", add=False)

        filesink = self._create_element("filesink", "filesink", "Sink", add=False)
        filesink.set_property("location", self.output_video_path)
        filesink.set_property("sync", 0)
        filesink.set_property("async", 0)

        # Gst.Bin.add(mp4_sink_bin, nvvidconv3)
        mp4_sink_bin.add(nvvidconv3)
        mp4_sink_bin.add(capsfilter2)
        mp4_sink_bin.add(encoder)
        mp4_sink_bin.add(codeparser)
        mp4_sink_bin.add(container)
        mp4_sink_bin.add(filesink)

        mp4_sink_bin.add_pad(Gst.GhostPad("sink", nvvidconv3.get_static_pad("sink")))
        self._link_sequential([nvvidconv3, capsfilter2, encoder, codeparser, container, filesink])
        self._add_element(mp4_sink_bin)

        return mp4_sink_bin

    def _create_rtsp_sink_bin(self):
        rtsp_sink_bin = Gst.Bin.new("rtsp-sink-bin")

        nvvidconv3 = self._create_element("nvvideoconvert", "convertor3", "Converter 3", add=False)
        capsfilter2 = self._create_element("capsfilter", "capsfilter2", "Caps filter 2", add=False)
        capsfilter2.set_property("caps",
                                 Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

        if self.rtsp_codec not in ["H264", "H265"]:
            raise ValueError(f"Invalid codec '{self.rtsp_codec}'")

        # Make the encoder
        encoder = self._create_element(f"nvv4l2{self.rtsp_codec.lower()}enc", "encoder",
                                       f"{self.rtsp_codec} encoder", add=False)
        encoder.set_property('bitrate', 4000000)

        if is_aarch64():
            encoder.set_property('preset-level', 1)
            encoder.set_property('insert-sps-pps', 1)
            encoder.set_property('bufapi-version', 1)

        # Make the payload-encode video into RTP packets
        rtppay = self._create_element(f"rtp{self.rtsp_codec.lower()}pay", "rtppay",
                                      f"{self.rtsp_codec} rtppay", add=False)

        # Make the UDP sink
        updsink_port_num = 5400
        sink = self._create_element("udpsink", "udpsink", "UDP sink", add=False)
        sink.set_property('host', '224.224.255.255')
        sink.set_property('port', updsink_port_num)
        sink.set_property('async', False)
        sink.set_property('sync', 1)

        rtsp_sink_bin.add(nvvidconv3)
        rtsp_sink_bin.add(capsfilter2)
        rtsp_sink_bin.add(encoder)
        rtsp_sink_bin.add(rtppay)
        rtsp_sink_bin.add(sink)

        rtsp_sink_bin.add_pad(Gst.GhostPad("sink", nvvidconv3.get_static_pad("sink")))
        self._link_sequential([nvvidconv3, capsfilter2, encoder, rtppay, sink])
        self._add_element(rtsp_sink_bin)

        return rtsp_sink_bin

    def _create_elements(self):
        self.source_bin = self._create_source_bin()
        self.streammux = self._create_streammux()

        self.pgie = self._create_element("nvinfer", "primary-inference", "PGIE")
        self.pgie.set_property('config-file-path', self.pgie_config_path)
        self.tracker = self._create_tracker()

        # Use convertor to convert from NV12 to RGBA (easier to work with in Python)
        self.nvvidconv1 = self._create_element("nvvideoconvert", "convertor1", "Converter 1")
        self.capsfilter1 = self._create_element("capsfilter", "capsfilter1", "Caps filter 1")
        self.capsfilter1.set_property("caps",
                                      Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        self.tiler = self._create_tiler()
        self.nvvidconv2 = self._create_element("nvvideoconvert", "convertor2", "Converter 2")

        if self.enable_osd:
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

    @staticmethod
    def _link_sequential(elements: list):
        for i in range(0, len(elements) - 1):
            elements[i].link(elements[i + 1])

    def _link_elements(self):
        self.logger.info(f"Linking elements in the Pipeline: {self}")

        sinkpad = self.streammux.get_request_pad("sink_0")
        if not sinkpad:
            self.logger.error("Unable to get the sink pad of streammux")
        srcpad = self.source_bin.get_static_pad("src")
        if not srcpad:
            self.logger.error("Unable to get source pad of decoder")
        srcpad.link(sinkpad)

        self._link_sequential(self.elements[1:])

    def _write_osd_analytics(self, batch_meta, l_frame_meta: List, ll_obj_meta: List[List]):
        obj_counter = defaultdict(int)

        for frame_meta, l_obj_meta in zip(l_frame_meta, ll_obj_meta):
            frame_number = frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta

            for obj_meta in l_obj_meta:
                obj_counter[obj_meta.class_id] += 1
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            py_nvosd_text_params.display_text = \
                "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(
                    frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE],
                    obj_counter[PGIE_CLASS_ID_PERSON])

            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            py_nvosd_text_params.set_bg_clr = 1
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

            self.logger.info(pyds.get_string(py_nvosd_text_params.display_text))
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            self.fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

    def _calculate_crop_score(self, track_id, crop):
        score = crop.size
        num_detections = len(self.track_scores[track_id])

        # Penalize entry frames
        if num_detections <= 15:
            score -= 1000

        return score

    def _save_crops(self, frames, _, l_frame_meta: List, ll_obj_meta: List[List]):
        self.logger.info(f"Saving crops to '{os.path.realpath(CROPS_DIR)}'")
        for frame, frame_meta, l_obj_meta in zip(frames, l_frame_meta, ll_obj_meta):
            frame_copy = np.array(frame, copy=True, order='C')
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)

            for obj_meta in l_obj_meta:
                track_id = obj_meta.object_id
                x1, y1, x2, y2 = rect_params_to_coords(obj_meta.rect_params)
                crop = frame_copy[y1:y2, x1:x2]
                crop_score = self._calculate_crop_score(track_id, crop)

                if not self.track_scores[track_id] or crop_score > max(self.track_scores[track_id]):
                    crop_dir = os.path.join(CROPS_DIR, f"src_{frame_meta.source_id}",
                                            f"obj_{obj_meta.object_id}_cls_{obj_meta.class_id}")
                    os.makedirs(crop_dir, exist_ok=True)
                    for f in os.listdir(crop_dir):
                        os.remove(os.path.join(crop_dir, f))
                    crop_path = os.path.join(crop_dir, f"frame_{frame_meta.frame_num}.jpg")
                    cv2.imwrite(crop_path, crop)
                    self.logger.debug(f"Saved crop to '{crop_path}'")

                self.track_scores[track_id].append(crop_score)

    def _probe_fn_wrapper(self, _, info, probe_fn, get_frames=False):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            self.logger.error("Unable to get GstBuffer")
            return

        frames = []
        l_frame_meta = []
        ll_obj_meta = []
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if get_frames:
                frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frames.append(frame)

            l_frame_meta.append(frame_meta)
            l_obj_meta = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                l_obj_meta.append(obj_meta)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            ll_obj_meta.append(l_obj_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        if get_frames:
            probe_fn(frames, batch_meta, l_frame_meta, ll_obj_meta)
        else:
            probe_fn(batch_meta, l_frame_meta, ll_obj_meta)

        return Gst.PadProbeReturn.OK

    def _wrap_probe(self, probe_fn):
        get_frames = "frames" in signature(probe_fn).parameters
        return partial(self._probe_fn_wrapper, probe_fn=probe_fn, get_frames=get_frames)

    @staticmethod
    def _get_static_pad(element, pad_name: str = "sink"):
        pad = element.get_static_pad(pad_name)
        if not pad:
            raise AttributeError(f"Unable to get {pad_name} pad of {element.name}")

        return pad

    def _add_probes(self):
        tiler_sinkpad = self._get_static_pad(self.tiler)

        if self.enable_osd and self.write_osd_analytics:
            tiler_sinkpad.add_probe(Gst.PadProbeType.BUFFER,
                                    self._wrap_probe(self._write_osd_analytics))

        if self.save_crops:
            tiler_sinkpad.add_probe(Gst.PadProbeType.BUFFER, self._wrap_probe(self._save_crops))

    def release(self):
        """Release resources and cleanup."""
        pass

    def run(self):
        # Create an event loop and feed gstreamer bus messages to it
        loop = GObject.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)

        if self.output_format == "rtsp":
            # Start streaming
            rtsp_port_num = 8554

            server = GstRtspServer.RTSPServer.new()
            server.props.service = "%d" % rtsp_port_num
            server.attach(None)

            factory = GstRtspServer.RTSPMediaFactory.new()
            factory.set_launch(
                "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (
                5400, self.rtsp_codec))
            factory.set_shared(True)
            server.get_mount_points().add_factory("/ds-test", factory)

            self.logger.info("\n *** DeepStream: Launched RTSP Streaming "
                             "at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)

        # Start play back and listen to events
        self.logger.info("Starting pipeline")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        except:
            pass

        self.logger.info("Exiting pipeline")
        self.pipeline.set_state(Gst.State.NULL)
        self.release()
