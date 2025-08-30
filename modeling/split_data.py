import os

import json

import numpy as np
import pandas as pd


def split_data(
    labels_dir,
    patch_dir,
    model_split_dir,
    seed:int|None=None,
    **kwargs
):
    if seed is not None:
        # set fixed seed
        np.random.seed(0)

    # get slide ids
    slide_ids = [entry.name.split('.')[0] for entry in os.scandir(labels_dir) 
        if entry.is_file() and entry.name.endswith(".csv")]


    # number of slides
    n = len(slide_ids)
    ds_size = {
        "train" : int(0.8 * n),
        "val"   : int(0.1 * n),
    }
    ds_size["test"] = n - ds_size["train"] - ds_size["val"]
    for typ in ds_size:
        print(f"{typ}: {ds_size[typ]} slides ")
    print("")

    slides = {
        "train" : slide_ids[                :ds_size["train"]],
        "val"   : slide_ids[ds_size["train"]:-ds_size["test"]],
        "test"  : slide_ids[-ds_size["test"]:                ]
    }

    # number of annots
    ds_annots = { "train": 0, "val": 0, "test": 0}
    for typ in slides:
        for slide_id in slides[typ]:
            df = pd.read_csv(f'{labels_dir}/{slide_id}.csv')
            ds_annots[typ] += len(df)
        print(f"{typ}: {ds_annots[typ]} annotations ")

    # patching
    ds_patches = {}
    ds_labels = {}
    for typ in slides:
        patches = []
        labels = []
        for slide_id in slides[typ]:
            df = pd.read_csv(f'{labels_dir}/{slide_id}.csv')
            slide_patches = np.load(f'{patch_dir}/{slide_id}.npy')[:, :, :, :3]
            labels.extend(df['class_id'])
            patches.extend(slide_patches)
        labels = 2 - np.array(labels)
        patches = np.array(patches)
        np.save(f'{model_split_dir}/{typ}_labels.npy', labels)
        np.save(f'{model_split_dir}/{typ}_patches.npy', labels)

    json.dump({
        'train_slides': slides['train'],
        'val_slides': slides['val'],
        'test_slides': slides['test']
    }, open(f'{model_split_dir}/slide_ids.json', 'w'), indent=4)

if __name__=="__main__":
    import yaml

    # load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    split_data(**config, seed=0)

