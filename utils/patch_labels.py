import os
from math import floor

import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.io import imread, imsave


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def patch_im(
    im:np.ndarray, 
    centers:pd.DataFrame,
    patch_size:int = config['patch_size'], 
    skip_boundary:bool=False
):  
    '''
    patches images around each pixel in centers

    im: slide stored as np.ndarray

    centers: dataframe of coordinates to center patches around

    patch_size: size of patches around cells

    skip_boundary: whether to skip cells on the boundary (i.e., cells whose patch
    will have white space)
    '''
    # parameters
    h, w, c = im.shape
    l = (patch_size - 1)//2
    u = patch_size//2

    # convert centroid coordinates to ints
    centers['x'], centers['y'] = centers['x'].apply(floor), centers['y'].apply(floor)

    # skip nuclei on boundary if skip_boundary
    if skip_boundary:
        x_mask = (centers['x'] - l >= 0) & (centers['x'] + u < w)
        y_mask = (centers['y'] - l >= 0) & (centers['y'] + u  < h)
        centers = centers[x_mask & y_mask]
    n = len(centers)
    xs, ys = np.array(centers['x']), np.array(centers['y'])
    
    patches = np.ones((n, patch_size, patch_size, c)) * 255

    for i in range(n):
        x, y = xs[i], ys[i]
        
        x_min, x_max = max(0, x - l), min(w, x + u + 1)
        y_min, y_max = max(0, y - l), min(h, y + u + 1)

        x_shift, y_shift = x - l, y - l

        # coordinate order y, x as plt.imshow default
        patches[i, y_min - y_shift:y_max - y_shift, x_min - x_shift:x_max - x_shift, :] = im[y_min:y_max, x_min:x_max, :]

    return patches


def patch_labels(
    labels_dir = config['labels_dir'], 
    image_dir = config['im_dir'], 
    patch_dir = config['patch_dir'], 
    patch_im_dir = config['patch_im_dir']
):
    slide_ids = [entry.name.split('.')[0] for entry in os.scandir(labels_dir) 
        if entry.is_file() and entry.name.endswith(".csv")]

    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(patch_im_dir, exist_ok=True)

    for slide_id in tqdm(slide_ids):
        df = pd.read_csv(f'{labels_dir}/{slide_id}.csv')
        im = imread(f'{image_dir}/{slide_id}.tiff')
        patches = patch_im(im, df)
        np.save(f'{patch_dir}/{slide_id}.npy', patches)

        for i in range(len(df)):
            os.makedirs(f'{patch_im_dir}/{slide_id}', exist_ok=True)
            imsave(
                f'{patch_im_dir}/{slide_id}/{df.annotation_id.loc[i]}.png', 
                patches[i].astype('uint8')
            )
    


if __name__ == '__main__':
    patch_labels()
