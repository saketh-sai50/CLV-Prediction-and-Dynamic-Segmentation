import yaml
from datetime import datetime
import os

def load_config(config_path="src/config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_watermark(watermark_path):
    """Reads the last run timestamp from the watermark file."""
    if os.path.exists(watermark_path):
        with open(watermark_path, 'r') as f:
            return datetime.fromisoformat(f.read().strip())
    # For the very first run, return a date in the past
    return datetime(2023, 1, 1)

def set_watermark(watermark_path, value):
    """Writes the current run timestamp to the watermark file."""
    with open(watermark_path, 'w') as f:
        f.write(value.isoformat())