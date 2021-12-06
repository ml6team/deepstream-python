import sys
from typing import List

import cv2

sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

from app.pipeline import Pipeline


class AnonymizationPipeline(Pipeline):

    def __init__(self, *args, target_classes: list = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_classes = target_classes

    @staticmethod
    def _anonymize_bbox(image, obj_meta, mode="pixelate"):
        rect_params = obj_meta.rect_params
        top = int(rect_params.top)
        left = int(rect_params.left)
        width = int(rect_params.width)
        height = int(rect_params.height)

        x1 = left
        y1 = top
        x2 = left + width
        y2 = top + height

        if mode == "blur":
            bbox = image[y1:y2, x1:x2]
            image[y1:y2, x1:x2] = cv2.GaussianBlur(bbox, (15, 15), 60)
        elif mode == "pixelate":
            reshape_factor = 18
            min_dim = 16
            bbox = image[y1:y2, x1:x2]
            h, w, _ = bbox.shape
            new_shape = (max(min_dim, int(w/reshape_factor)), max(min_dim, int(h/reshape_factor)))
            bbox = cv2.resize(bbox, new_shape, interpolation=cv2.INTER_LINEAR)
            image[y1:y2, x1:x2] = cv2.resize(bbox, (w, h), interpolation=cv2.INTER_NEAREST)
        elif mode == "fill":
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=-1)
        else:
            raise ValueError(f"Invalid anonymization mode '{mode}'.")

        return image

    def _anonymize(self, frames, _, l_frame_meta: List, ll_obj_meta: List[List]):
        for frame, frame_meta, l_obj_meta in zip(frames, l_frame_meta, ll_obj_meta):
            for obj_meta in l_obj_meta:
                if self.target_classes and obj_meta.class_id not in self.target_classes:
                    continue

                frame = self._anonymize_bbox(frame, obj_meta)

    def _add_probes(self):
        super()._add_probes()
        tiler_sinkpad = self._get_static_pad(self.tiler)
        tiler_sinkpad.add_probe(Gst.PadProbeType.BUFFER, self._wrap_probe(self._anonymize))
