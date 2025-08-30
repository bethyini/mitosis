import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def get_dataloaders(
    model_split_dir:str,
    batch_size:int,
    color_mean,
    color_std,
    **kwargs
):
    '''
    make dataloaders for training / validating data
    '''
    loaders = {}
    for group in ["train", "val", "test"]:
        loaders[group] = get_dataloader(
            f"{model_split_dir}/{group}_patches.npy",
            f"{model_split_dir}/{group}_labels.npy",
            batch_size,
            color_mean,
            color_std
        )

    return loaders


def get_dataloader(
    patches_path:str,
    labels_path:str|None = None,
    batch_size:int=64,
    color_mean=None,
    color_std=None
):
    '''
    get dataloader
    '''
    patches = np.load(patches_path) / 255
    if labels_path is not None:
        labels = np.load(labels_path)
    else: 
        labels = Noen
    dataset = NumpyDataset(patches, labels, color_mean, color_std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class NumpyDataset(Dataset):
    def __init__(
        self,
        X:np.ndarray,
        y:np.ndarray|None = None,
        color_mean=None,
        color_std=None
    ):
        """
        X : numpy array of shape (N, H, W, C) or (N, C, H, W)
        y : numpy array of shape (N,)
        """
        
        self.X = X
        self.y = y
        if color_mean is None:
            color_mean = [0.5, 0.5, 0.5]
        if color_std is None:
            color_std = [0.5, 0.5, 0.5]
        self.transform = transforms.Normalize(mean=color_mean, std=color_std)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]

        # convert numpy to tensor (and reorder channels if needed)
        if img.ndim == 3 and img.shape[-1] in [1, 3]:   # assume NHWC
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # to CHW
        else:
            img = torch.from_numpy(img).float()

        # apply optional transforms (e.g. normalization)
        if self.transform:
            img = self.transform(img)
        
        if self.y is not None:
            label = torch.tensor(self.y[idx]).long()  # class index
            return img, label
        else:
            return img