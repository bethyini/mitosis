import os
import numpy as np
from skimage.io import imread, imsave
import torchstain
from tqdm import tqdm

def normalize_color(
    im,
    target,
):
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(target)
    normalized, _, _ = normalizer.normalize(im)
    return normalized

def normalize_slides(
    target_im,
    original_image_dir,
    image_dir,
    **kwargs
):
    slide_ids = [entry.name.split('.')[0] for entry in os.scandir(original_image_dir) 
        if entry.is_file() and entry.name.endswith(".tiff")]

    os.makedirs(image_dir, exist_ok=True)
    
    for im_idx in tqdm(slide_ids):
        target = imread(target_im)[:, :, :3]
        im = imread(f"{original_image_dir}/{im_idx}.tiff")[:, :, :3]
        normalized = normalize_color(im, target)
        imsave(
            f'{image_dir}/{im_idx}.tiff', 
            normalized.astype('uint8')
        )

if __name__=="__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    normalize_slides(**config)