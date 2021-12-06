import json
import os
import sys
from collections import defaultdict

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
import ctypes
import numpy as np

from app.pipeline import Pipeline
from app.config import CONFIGS_DIR, OUTPUT_DIR


class ReIDPipeline(Pipeline):

    def __init__(self, *args,
                 sgie_config_path: str = os.path.join(CONFIGS_DIR, "sgies/osnet.txt"),
                 target_classes: list = None, **kwargs):
        self.sgie_config_path = sgie_config_path
        self.target_classes = target_classes
        self.sgie = None
        self.reid_features = defaultdict(list)
        self.json_path = os.path.join(OUTPUT_DIR, "reid_features.json")

        super().__init__(*args, **kwargs)

    def _create_elements(self):
        super()._create_elements()
        element_names = [elm.name for elm in self.elements]
        tracker_idx = element_names.index(self.tracker.name)

        self.sgie = self._create_element("nvinfer", "secondary-inference", "SGIE", add=False)
        self.sgie.set_property('config-file-path', self.sgie_config_path)
        self._add_element(self.sgie, tracker_idx + 1)

    def _save_features(self, _, __, ll_obj_meta):
        for l_obj_meta in ll_obj_meta:
            for obj_meta in l_obj_meta:
                l_user = obj_meta.obj_user_meta_list
                while l_user is not None:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    except StopIteration:
                        break

                    if user_meta.base_meta.meta_type \
                            != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        continue

                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer),
                                      ctypes.POINTER(ctypes.c_float))
                    features = np.ctypeslib.as_array(ptr, shape=(512,))
                    self.reid_features[obj_meta.object_id].append(features.tolist())

                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

    def _add_probes(self):
        super()._add_probes()
        sgie_src_pad = self._get_static_pad(self.sgie, "src")
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._wrap_probe(self._save_features))

    def release(self):
        self.logger.info(f"Saving ReID features to '{os.path.realpath(self.json_path)}'")
        with open(self.json_path, "w") as out_file:
            json.dump(self.reid_features, out_file, indent=2)
