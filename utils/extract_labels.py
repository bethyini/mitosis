import os
import argparse
import sqlite3

import yaml
from tqdm import tqdm
import pandas as pd


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def annotations_to_by_slide(sqlite_path):
    """Get sample annotations with coordinates and labels"""

    # change to directory of annotations
    os.chdir('/'.join(sqlite_path.split('/')[:-1]))
    db_name = sqlite_path.split('/')[-1]

    # connect to sqlite file
    conn = sqlite3.connect(db_name)
    
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

    # change to directory outside annotation directory
    os.chdir('..')
    # create directory for labels
    os.makedirs(config['labels_dir'], exist_ok=True)

    # split dataframe by slide
    for slide_file in tqdm(set(df.slide_file)):
        df_slide = df[df.slide_file==slide_file]
        df_slide = df_slide.drop(columns='slide_file')
        df_slide.to_csv(f'{config["labels_dir"]}/{slide_file.split(".")[0]}.csv', index=False)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("sqlite_path", help="Path to sqlite file of annotations")
    # args = parser.parse_args()
    # annotations_to_by_slide(args.sqlite_path)
    annotations_to_by_slide(config['annot_path'])


