import os
import re
import pandas as pd


def files_match_pattern(file_dir, pattern):
    file_list = []
    for f in os.listdir(file_dir):
        if re.search(pattern,f):
            file_list.append(f)
    return file_list


def decode_files(file_list):
    classes = set()
    groups = set()
    samples = set()
    for f in file_list:
        f = f.split('.')[0]
        classes.add(f.split('_')[0])
        groups.add(f.split('_')[1])
        samples.add(f.split('_')[2])
    return list(classes), list(groups), list(samples)


def read_sep_file(file):
    # Determine separator based on file extension
    _, ext = os.path.splitext(file)
    ext = ext.lower()

    if ext == ".csv":
        df = pd.read_csv(file, delimiter=',')
    elif ext == ".tsv":
        df = pd.read_csv(file, delimiter='\t')
    else:
        df = None
    return df