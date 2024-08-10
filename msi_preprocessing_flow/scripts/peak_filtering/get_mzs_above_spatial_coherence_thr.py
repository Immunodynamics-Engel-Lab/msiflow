import argparse
import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def merge_files(base_df, file_dir, file_list):
    for fl in file_list:
        fl_df = pd.read_csv(os.path.join(file_dir, fl), index_col=0)
        fl_name = fl.split('_')[0] + '_' + fl.split('_')[1]
        fl_df.rename(columns={"Spatial coherence": fl_name}, inplace=True)
        base_df = pd.merge(base_df, fl_df, left_index=True, right_index=True)
    return base_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate overall spatial coherence and extract m/z above defined threshold')
    parser.add_argument('dir', type=str, help='directory with spatial coherence files')
    parser.add_argument('imzML_dir', type=str, help='directory with imzML files')
    parser.add_argument('out_dir', type=str, help='output directory')
    parser.add_argument('-thr', type=int, default=100, help='spatial coherence threshold')
    parser.add_argument('-sum', type=str, default='mean', help='how to summarize over all sc values')
    args = parser.parse_args()

    # read in data
    imzML_files = [file for file in os.listdir(args.imzML_dir) if file.endswith('.imzML')]
    #imzML_files = np.array(sorted(imzML_files, key=lambda x: int(x.split('.')[0][-2:])))
    sc_files = [file for file in os.listdir(args.dir) if file.endswith('_sc.csv')]

    df = pd.read_csv(os.path.join(args.dir, sc_files[0]))
    mzs = df.iloc[:, 0].to_numpy()

    sc_df = pd.DataFrame(index=mzs)
    sc_df = merge_files(sc_df, args.dir, sc_files)

    # get mean spatial coherence of all samples and sort accordingly
    if args.sum == 'min':
        sc_df[args.sum] = sc_df.min(axis=1)
    elif args.sum == 'mean':
        sc_df[args.sum] = sc_df.mean(axis=1)
    else:
        sc_df[args.sum] = sc_df.max(axis=1)
    sc_df = sc_df.sort_values(by=args.sum, ascending=False)

    # get m/z above threshold
    filtered_sc_df = sc_df[sc_df[args.sum] > args.thr]
    mzs_above_thr = filtered_sc_df.index.to_numpy()

    sc_df.to_csv(os.path.join(args.out_dir, 'overall_spatial_coherence.csv'))
    #np.save(os.path.join(args.out_dir, 'peaks_above_' + str(args.thr) + '_sc_' + str(mzs_above_thr.shape[0]) + 'bins.npy'), mzs_above_thr)
    np.save(os.path.join(args.out_dir, 'peaks_above_' + str(args.thr) + 'sc.npy'), mzs_above_thr)
