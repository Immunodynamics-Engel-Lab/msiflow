import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import pandas as pd

import tifffile
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm import tqdm
from scipy.stats.mstats import pearsonr

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils


def get_mz_img(pyx, msi_df, mz):
    coords = msi_df.index.tolist()
    msi_img = np.zeros(pyx).astype(np.uint8)
    for x_val, y_val in coords:
        msi_img[y_val - 1, x_val - 1] = msi_df.loc[(x_val, y_val), mz]
    return msi_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts matrix peaks and tissue peaks based on spatial correlation'
                                                 'to matrix region')
    parser.add_argument('imzML_fl', type=str, help='imzML file')
    parser.add_argument('matrix_img', type=str, help='tif file with matrix region')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result')
    parser.add_argument('-method', type=str, default='median', help='method for normalization')
    parser.add_argument('-debug', type=bool, default=False, help='set to True for debugging')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "matrixpeaks")
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)

    p = ImzMLParser(args.imzML_fl)
    pyx = (p.imzmldict["max count of pixels y"], p.imzmldict["max count of pixels x"])
    df = utils.get_dataframe_from_imzML(args.imzML_fl, multi_index=True)
    mzs = df.columns.to_numpy()
    matrix_img = utils.NormalizeData(tifffile.imread(args.matrix_img))
    matrix_img[matrix_img < 1] = 0

    # get correlation for each m/z value
    pearson = []
    for i, mz in enumerate(tqdm(mzs)):
        ion_img = utils.NormalizeData(get_mz_img(pyx, df, mz))
        pearson.append(pearsonr(x=matrix_img.ravel(), y=ion_img.ravel())[0])

    df_corr = pd.DataFrame(data=pearson, columns=['Correlation'], index=mzs)
    df_corr.to_csv(os.path.join(args.result_dir, 'corr_' + os.path.basename(args.imzML_fl).split('.')[0] + '.csv'))