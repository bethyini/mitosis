import yaml

from utils import *
from modeling import *

import numpy as np
import torch

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


if __name__ == '__main__':
    # PROCESSING DATA
    # extract_labels(**config)
    # patch_labels(**config)

    np.random.seed(0)
    torch.manual_seed(0)

    # MODEL TRAINING
    train_loader, val_loader, _ = get_dataloaders(**config)
    # model = CNNClassifer()
    model = DenseNet(
        depth=40, 
        num_classes=2, 
        growth_rate=12, 
        reduction=0.5, 
        dropRate=config['dropout'], 
        bottleneck=True
    )
    model = train(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=config["num_epochs"], 
        lr=config["lr"], 
        lmbda_l2=config["lmbda_l2"],
        device=config["device"]
    )