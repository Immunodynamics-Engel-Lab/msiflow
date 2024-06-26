import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg.utils import get_mean_spectrum, get_summarized_spectrum, get_dataframe_from_imzML, get_combined_dataframe_from_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts median overall spectrum')
    parser.add_argument('imzML_dir', type=str, help='folder to MSI imzML files')
    parser.add_argument('output', type=str, help='output file')
    args = parser.parse_args()

    files = [f for f in os.listdir(args.imzML_dir) if os.path.isfile(os.path.join(args.imzML_dir, f))
             and f.endswith('.imzML')]
    df = get_combined_dataframe_from_files(args.imzML_dir, files, multi_index=True)

    df = df.replace(0, np.nan)
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)

    data = df.to_numpy()
    median_spec = np.nanmedian(data, axis=0)

    np.save(args.output, median_spec)
