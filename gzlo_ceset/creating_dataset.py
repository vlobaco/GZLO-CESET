
from gzlo_ceset import utils 
from typing import List
import argparse
import os
import pandas as pd
import tqdm

def file_to_text(path: str) -> List:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    partition = utils.partition_file(path)
    return utils.mine_text_from_partition(partition)

def folder_to_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Folder not found: {path}")
    data = []
    for root, _, files in os.walk(path):
        components = root[len(path) + 1:].split(os.sep)
        if len(components) < 2 or len(files) == 0:
            continue
        area = components[0]
        doc_type = components[1]
        if len(components) > 2:
            operation = components[2]
        else:
            operation = area
        for file in tqdm.tqdm(files, desc=f'{area}/{doc_type}/{operation}'):
            try:
                raw_text = file_to_text(os.path.join(root, file))
            except:
                print(f"Error reading file: {file}")
                continue
            tokens = utils.preprocess_text(raw_text, remove_stop_words=True, remove_punctuation=True, remove_accents=True, to_lower=True)
            data += [{
                "file_name": file,
                "area": area,
                "doc_type": doc_type,
                "operation": operation,
                "raw_text": raw_text,
                "tokens": tokens,
                "clean_text": ' '.join(tokens)
            }]
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Reads a folder with text files and creates a DataFrame with the content of the files.\n' +
            'The folder must have the structure: <area>/<doc_type>/<operation>',
        usage='python creating_dataset.py --path <path> --dest <dest.pkl>'
        )
    parser.add_argument('--path', type=str, required=True, help='The path to the files')
    parser.add_argument('--dest', type=str, required=True, help='The destination for the output')

    args = parser.parse_args()

    path = args.path
    dest = args.dest
    df = folder_to_df(path)
    df.to_pickle(dest)
    print(f"DataFrame saved to {dest}")