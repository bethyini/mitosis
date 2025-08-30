from modeling import *

def train_model(
    **kwargs
):
    if "seed" in kwargs and kwargs["seed"] is not None:
        import torch
        import numpy as np
        np.random.seed(kwargs["seed"])
        torch.manual_seed(kwargs["seed"])

    loaders = get_dataloaders(**kwargs)

    model = DenseNet(
        num_classes=2, 
        **kwargs
    )

    model = train(
        model, 
        loaders['train'], 
        loaders['val'],
        **kwargs
    )


if __name__ == "__main__":

    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    train_model(**config)