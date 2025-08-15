import yaml

from utils import *

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


if __name__ == '__main__':
    extract_labels(**config)
    patch_labels(**config)