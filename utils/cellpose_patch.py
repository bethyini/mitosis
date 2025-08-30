import os
from math import floor
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.io import imread, imsave # for reading/writing image
from skimage.measure import regionprops # centroid finding

from cellpose import models, utils

try:
    from .patch_labels import patch_im
except:
    from patch_labels import patch_im


def cellpose_patch(
    im_idx,
    image_dir:str,
    patch_size:int,
    save_cell_patch_ims:bool=False,
    cell_dir:str="cells",
    cell_masks_dir:str="masks",
    cell_patch_im_dir:str="patch_ims",
    cell_outlines_dir:str="outlines",
    cell_centers_dir:str="centers",
    skip_boundary=False,
    **kwargs
):
    '''
    run cellpose on slides

    patch_size: size of patches around cells

    skip_boundary: whether to skip cells on the boundary (i.e., cells whose patch
    will have white space)
    '''

    os.makedirs(cell_masks_dir, exist_ok=True)
    os.makedirs(cell_outlines_dir, exist_ok=True)
    os.makedirs(cell_centers_dir, exist_ok=True)
    os.makedirs(cell_dir, exist_ok=True)

    masks_path = f"{cell_masks_dir}/{im_idx}.npy"
    centers_path = f"{cell_centers_dir}/{im_idx}.csv"
    outlines_path = f"{cell_outlines_dir}/{im_idx}.pkl"
    patches_path = f"{cell_dir}/{im_idx}.npy"

    # Load image
    im = imread(f"{image_dir}/{im_idx}.tiff")
    # Remove alpha channel if exists:
    if im.shape[2] == 4:
        im = im[:, :, :3]
    
    # get nuclei masks of slide
    if not os.path.isfile(masks_path):
        masks = get_nuceli_mask(im)
        np.save(masks_path, masks)
    else:
        masks = np.load(masks_path)
    
    # get outlines of nuclei
    if not os.path.isfile(outlines_path):
        outlines = utils.outlines_list(masks)
        pickle.dump(outlines, open(outlines_path, "wb"))

    # get centroids of nuclei
    if not os.path.isfile(centers_path):
        centroids = compute_centroids(masks)
        centroids.to_csv(centers_path, index=False)
    else:
        centroids = pd.read_csv(centers_path)

    # get cell patches
    if not os.path.isfile(patches_path):
        patches = patch_im(im, centroids, patch_size, skip_boundary=skip_boundary)
        np.save(patches_path, patches)
    else:
        patches = np.load(patches_path)

    # save as images
    if save_cell_patch_ims:
        print('saving patches as images...')
        os.makedirs(f'{cell_patch_im_dir}/{im_idx}/', exist_ok=True)
        for i in tqdm(range(len(patches))):
            imsave(f'{cell_patch_im_dir}/{im_idx}/{i}.png', patches[i].astype('uint8'))

    return patches


def get_nuceli_mask(
    im:np.ndarray,
):
    ''' 
    get nuceli mask using cellpose 
    ''' 
    
    # Remove alpha channel if exists:
    if im.shape[2] == 4:
        im = im[:, :, :3]

    # Load model (automatically select gpu if available)
    model = models.CellposeModel(gpu=True)

    # Evaluate
    masks, flows, styles = model.eval(
        im, diameter=None, # Automatically estimate diameter
    )

    return masks



def compute_centroids(
    masks:np.ndarray,
):
    '''
    find nuceli centroids from nuclei mask
    '''

    props = regionprops(masks)
    centroids = np.array([p.centroid for p in props])

    return pd.DataFrame(centroids, columns=['y', 'x'])


if __name__=="__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    import json
    split = json.load(open("data/model_split/slide_ids.json", "r"))

    from tqdm import tqdm

    for slide_id in tqdm(split["test_slides"]):
        cellpose_patch(slide_id, **config)

