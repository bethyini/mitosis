import os
from math import floor

import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import rescale


def patch_im(
    im:np.ndarray, 
    centers:pd.DataFrame,
    patch_size, 
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


def resize_im(
   im:np.ndarray,
   scale:float
):
    return rescale(
        im, 
        scale=scale,
        channel_axis=-1,
        anti_aliasing=True
    ) * 255


def patch_labels(
    labels_dir:str,
    image_dir:str,
    patch_dir:str,
    patch_size:int,
    save_patch_ims:bool=False,
    patch_ims_dir:str|None=None,
    resize:bool=False,
    resized_dir:str|None=None,
    magn:float|None=None,
    **kwargs
):
    '''
    get patches of each slide around each annotated label 
    '''
    slide_ids = [entry.name.split('.')[0] for entry in os.scandir(labels_dir) 
        if entry.is_file() and entry.name.endswith(".csv")]
    # for debug
    # slide_ids = ['001', '002', '003']

    slide_dir = image_dir

    # resize images if different magnification is desired
    if resize:
        scale = magn / 40
        os.makedirs(resized_dir, exist_ok=True)
        print('Resizing...')
        for slide_id in tqdm(slide_ids):
            im = imread(f'{slide_dir}/{slide_id}.tiff')
            resized_im = resize_im(im, scale)
            imsave(
                f'{resized_dir}/{slide_id}.tiff', 
                resized_im.astype('uint8')
            )
        slide_dir = resized_dir

    os.makedirs(patch_dir, exist_ok=True)
    if save_patch_ims:
        os.makedirs(patch_ims_dir, exist_ok=True)

    print('Patching...')
    for slide_id in tqdm(slide_ids):
        df = pd.read_csv(f'{labels_dir}/{slide_id}.csv')
        if resize: # scale coordinates as well 
            df['x'], df['y'] = df['x'] * scale, df['y'] * scale 
        im = imread(f'{slide_dir}/{slide_id}.tiff')
        patches = patch_im(im, df, patch_size)
        np.save(f'{patch_dir}/{slide_id}.npy', patches)

        if not save_patch_ims: continue
        # save patches as .png files
        for i in range(len(df)):
            os.makedirs(f'{patch_ims_dir}/{slide_id}', exist_ok=True)
            imsave(
                f'{patch_ims_dir}/{slide_id}/{df.annotation_id.loc[i]}.png', 
                patches[i].astype('uint8')
            )


if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    patch_labels(**config)
