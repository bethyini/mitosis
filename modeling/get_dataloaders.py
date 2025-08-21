import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

### NOTE TO SELF ###############################################################
# TODO: consider image normalization 
# stain tools (https://github.com/Peter554/StainTools) has WSI specific
# normalization x - mu / sigma normalization recommended (mu = sigma = 0.5 can
# be used as a started but dataset specific constants are recommended)
################################################################################

def get_dataloaders(
    train_patches_path:str,
    train_labels_path:str,
    val_patches_path:str,
    val_labels_path:str,
    test_patches_path:str,
    test_labels_path:str,
    batch_size:int,
    **kwargs
):
    '''
    make dataloaders for training / validating data
    '''
    train_loader = get_dataloader(train_patches_path, train_labels_path, batch_size)
    val_loader = get_dataloader(val_patches_path, val_labels_path, batch_size)
    test_loader = get_dataloader(test_patches_path, test_labels_path, batch_size)
    return train_loader, val_loader, test_loader


def get_dataloader(
    patches_path:str,
    labels_path:str,
    batch_size:int
):
    '''
    get dataloader
    '''
    patches = np.load(patches_path) / 255
    labels = np.load(labels_path)
    dataset = NumpyDataset(patches, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



transform = transforms.Normalize(mean=(0.693, 0.417, 0.702), std=(0.201, 0.226, 0.164))

class NumpyDataset(Dataset):
    def __init__(
        self,
        X:np.ndarray,
        y:np.ndarray,
        transform=transform
    ):
        """
        X : numpy array of shape (N, H, W, C) or (N, C, H, W)
        y : numpy array of shape (N,)
        """
        
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]

        # convert numpy to tensor (and reorder channels if needed)
        if img.ndim == 3 and img.shape[-1] in [1, 3]:   # assume NHWC
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # to CHW
        else:
            img = torch.from_numpy(img).float()

        label = torch.tensor(self.y[idx]).long()  # class index

        # apply optional transforms (e.g. normalization)
        if self.transform:
            img = self.transform(img)

        return img, label