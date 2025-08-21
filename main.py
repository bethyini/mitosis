import yaml

from utils import *
from modeling import *

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


if __name__ == '__main__':
    # PROCESSING DATA
    # extract_labels(**config)
    # patch_labels(**config)

    # MODEL TRAINING
    train_loader, val_loader, _ = get_dataloaders(**config)
    model = CNNClassifer()
    model = train(model, train_loader, val_loader, num_epochs=50, lr=5e-4)
