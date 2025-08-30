import yaml
from train_model import train_model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# PROCESSING DATA
# extract_labels(**config)
# patch_labels(**config)

# MODEL TRAINING
train_model(**config)