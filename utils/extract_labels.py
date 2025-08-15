import os
import argparse
import sqlite3

import yaml
from tqdm import tqdm
import pandas as pd


def extract_labels(**kwargs):
    """Get sample annotations with coordinates and labels"""

    # connect to sqlite file
    conn = sqlite3.connect(kwargs['annot_path'])
    
    query = """
    SELECT 
        a.uid as annotation_id,
        s.filename as slide_file,
        ac.coordinateX as x,
        ac.coordinateY as y,
        a.agreedClass as class_id,
        c.name as class_name
    FROM Annotations a
    JOIN Annotations_coordinates ac ON a.uid = ac.annoId
    JOIN Slides s ON a.slide = s.uid  
    JOIN Classes c ON a.agreedClass = c.uid
    WHERE a.deleted = 0
    """
    
    # create pandas.DataFrame
    df = pd.read_sql(query, conn)
    conn.close()

    # create directory for labels
    os.makedirs(kwargs['labels_dir'], exist_ok=True)

    # split dataframe by slide
    for slide_file in tqdm(set(df.slide_file)):
        df_slide = df[df.slide_file==slide_file]
        df_slide = df_slide.drop(columns='slide_file')
        df_slide.to_csv(f'{kwargs["labels_dir"]}/{slide_file.split(".")[0]}.csv', index=False)


if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    extract_labels(**config)


