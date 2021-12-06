import os

DEEPSTREAM_DIR = "/opt/nvidia/deepstream/deepstream-6.0"
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO")
PATH = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(PATH, "..", "configs")
DATA_DIR = os.path.join(PATH, "..", "data")
OUTPUT_DIR = os.path.join(PATH, "..", "output")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
