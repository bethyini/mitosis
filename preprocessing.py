from utils import *

def preprocessing(**kwargs):
    extract_labels(**kwargs)
    patch_labels(**kwargs)

if __name__ == "__main__":

    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    preprocessing(**config)