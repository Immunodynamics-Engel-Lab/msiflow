import argparse
import os
import sys
import pandas as pd
from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np

sys.path.append("..")
from pkg import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter out m/z values based on spatial coherence')
    parser.add_argument('imzML', type=str, help='imzML_file')
    # parser.add_argument('sc_file', type=str, help='file with spatial coherence')
    parser.add_argument('sc_thr', type=int, help='threshold to filter')
    parser.add_argument('ref_mz', type=str, help='file with numpy mz')
    parser.add_argument('-result_dir', type=str, default='', help='directory to save sc filtered file')
    args = parser.parse_args()

    # read in data
    # sc_df = pd.read_csv(args.sc_file,delimiter=',')
    # print(sc_df)

    # get mzs above sc threshold
    #filtered_sc_df = sc_df[sc_df['Spatial coherence'] > args.sc_thr]
    #filtered_sc_df = sc_df[sc_df['mean'] < args.sc_thr]
    #filtered_sc_df = sc_df
    #mzs_above_thr = filtered_sc_df[filtered_sc_df.columns[0]].to_numpy().astype(np.float32)
    #print(filtered_sc_df)
    #mzs_above_thr = sc_df[sc_df.columns[0]].to_numpy().astype(np.float32)
    #print(mzs_above_thr.shape)
    mzs_above_thr = np.load(args.ref_mz).astype(np.float32)
    print(mzs_above_thr)

    if args.result_dir == '':
        print("sc_filtered_" + str(mzs_above_thr.shape[0]))
        args.result_dir = os.path.join(os.path.dirname(args.imzML), "sc_filtered_" + str(args.sc_thr) + '_thr' +
                                       str(mzs_above_thr.shape[0]) + '_peaks')
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # mzs_above_thr = np.load(args.ref_mz).astype(np.float32)
    # print(mzs_above_thr.shape)

    # reduce data to defined mz
    p = ImzMLParser(args.imzML)
    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML))) as writer:
        for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
            mzs, intensities = p.getspectrum(idx)
            mzs = mzs.astype(np.float32)
            peaks_idx = np.where(np.in1d(mzs, mzs_above_thr))[0]
            writer.addSpectrum(mzs[peaks_idx], intensities[peaks_idx], (x, y, z))

