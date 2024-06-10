import numpy as np
import pandas as pd
import os
import argparse
import warnings
import sys
from skimage.exposure import equalize_adapthist, rescale_intensity

sys.path.append("..")
from pkg import utils
from pkg.plot import plot_mz_umap

warnings.filterwarnings('ignore', module='pyimzml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots intensity of m/z value in defined UMAP of multiple samples')
    parser.add_argument('imzML_dir', type=str, help='directory with imzML files')
    parser.add_argument('umap_file', type=str, default='', help='file with UMAP embedding')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('-mz', type=float, default=None, help='mz value')
    parser.add_argument('-mz_file', type=str, default='', help='csv file with m/z values')
    parser.add_argument('-num', type=int, default=None, help='takes this number of top m/z values from file to save')
    parser.add_argument('-contrast_stretch', type=bool, default=False, help='set to True to perform contrast stretch on each sample')
    parser.add_argument('-cmap', type=str, default='inferno', help='cmap to use for all plots')
    parser.add_argument('-dot_size', type=float, default=1, help='size for dots in scatterplots')
    parser.add_argument('-pos_group', type=str, default='control', help='name of control group')
    parser.add_argument('-neg_group', type=str, default='UPEC', help='name of infected group')
    parser.add_argument('-show_neg_group_only', type=bool, default=False, help='set to true to only plot negative group')
    parser.add_argument('-plot', type=bool, default=False, help='set to true if output should be plotted')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df_umap_clusters = pd.read_csv(args.umap_file)
    df_umap_clusters = df_umap_clusters[['group', 'sample', 'UMAP_1', 'UMAP_2', 'x', 'y']]

    imzML_files = [f for f in os.listdir(args.imzML_dir) if f.endswith('.imzML') and not f.startswith('.')
                   and os.path.isfile(os.path.join(args.imzML_dir, f))]
    df_msi = utils.get_combined_dataframe_from_files(args.imzML_dir, imzML_files, groups=True)

    # merge dataframes
    df = pd.merge(df_umap_clusters, df_msi, on=["group", "sample", "x", "y"], how='inner')
    #df = df[df['label'] != -1]
    #df['label'] = df['label'].astype('int')
    #df.sort_values(['label'], inplace=True)
    #df.set_index('label', inplace=True)

    mz_cols_round = np.round(df.columns[6:].astype(float).to_numpy(), 4).tolist()
    meta_cols = df.columns[:6].to_list()
    cols = meta_cols + mz_cols_round
    df.columns = cols

    df_control = None
    if args.show_neg_group_only:
        df_control = df[df['group'] == args.pos_group]
        df_control = df_control[['UMAP_1', 'UMAP_2', args.mz]]
        df_upec = df[df['group'] == args.neg_group]

        if args.contrast_stretch:
            df_scaled = pd.DataFrame(columns=['UMAP_1', 'UMAP_2', args.mz])
            for smpl in np.unique(df_upec['sample'].to_numpy()):
                df_smpl = df_upec[df_upec['sample'] == smpl]
                # print(df_smpl)
                low, up = np.percentile(df_smpl[args.mz].to_numpy(), (0, 99.9))
                rescaled = rescale_intensity(df_smpl[args.mz].to_numpy(), in_range=(low, up))
                df_smpl_rescaled = pd.DataFrame.from_dict(
                    {'UMAP_1': df_smpl['UMAP_1'].to_numpy(), 'UMAP_2': df_smpl['UMAP_2'],
                     args.mz: rescaled})
                # print(df_smpl_rescaled)
                df_scaled = df_scaled.append(df_smpl_rescaled)
                # print(df_scaled)
            # low, up = np.percentile(df[args.mz].to_numpy(), (0, 99.9))
            # df[args.mz] = rescale_intensity(df[args.mz].to_numpy(), in_range=(low, up))
            # df = df_scaled.append(df_control)
            df = df_scaled
        else:
            df = df_upec

    if args.mz:
        args.mz = np.round(args.mz, 4)
        output_file = os.path.join(args.output_dir, str(args.mz) + '_umap.png')
        plot_mz_umap(df, args.mz, output_file, df_control, args.show_neg_group_only, args.cmap, args.dot_size,
                     args.plot)
    else:
        df_mz_file = pd.read_csv(args.mz_file)
        if args.num:
            df_mz_file = df_mz_file.head(args.num)
        mz_arr = df_mz_file[df_mz_file.columns[0]].to_numpy().astype(float)
        mz_arr = np.round(mz_arr, 4)
        for mz in mz_arr:
            output_file = os.path.join(args.output_dir, str(mz) + '_umap.png')
            plot_mz_umap(df, mz, output_file, df_control, args.show_neg_group_only, args.cmap, args.dot_size,
                         args.plot)






