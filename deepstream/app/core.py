import os
import logging

from app.pipelines import Pipeline, AnonymizationPipeline, ReIDPipeline, AnalyticsPipeline, \
    SegmentationPipeline
from app.config import CONFIGS_DIR, LOGLEVEL

logging.basicConfig(level=LOGLEVEL)


def run_pipeline(video_uri: str):
    pipeline = Pipeline(
        video_uri=video_uri,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/pgie.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        output_format="mp4",
        # input_shape=(640, 360),
    )
    pipeline.run()


def run_segmentation_pipeline(video_uri: str):
    pipeline = SegmentationPipeline(
        video_uri=video_uri,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/segmentation.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/klt.txt"),
        output_format="mp4",
    )
    pipeline.run()


def run_anonymization_pipeline(video_uri: str):
    pipeline = AnonymizationPipeline(
        video_uri=video_uri,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/yolov4.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt"),
        target_classes=[2],
        enable_osd=False,
    )
    pipeline.run()


def run_reid_pipeline(video_uri: str):
    pipeline = ReIDPipeline(
        video_uri=video_uri,
        target_classes=[0],
        save_crops=False,
    )
    pipeline.run()


def run_analytics_pipeline(video_uri: str):
    pipeline = AnalyticsPipeline(
        video_uri=video_uri,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/yolov4-tiny.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/klt.txt"),
        conn_str="127.0.0.1;9092;test",
        schema_type=1,
    )
    pipeline.run()
