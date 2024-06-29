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
from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils
from pkg.plot import get_mz_img


def get_spectrum_from_bin_img(img, msi_df):
    bin_img_px_idx_np = np.nonzero(img)
    pixels_df = pd.DataFrame.from_dict({'x': bin_img_px_idx_np[1], 'y': bin_img_px_idx_np[0]})

    # get spectrum
    df = pd.merge(left=msi_df, right=pixels_df, on=['x', 'y'])
    spec = df.iloc[:, 2:].median(axis=0).to_numpy()
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get matrix pixels from segmentation and generated clusters '
                                                 'including matrix cluster (connected by most pixels to the tissue broder)')
    parser.add_argument('imzML_fl', type=str, help='imzML file')
    parser.add_argument('img_dir', type=str, help='directory to binary images of clusters including matrix and tissue image')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result, default=\'\' will create a directory')
    parser.add_argument('-matrix_corr_thr', type=float, default=0.7, help='min correlation to matrix, default=0.7')
    parser.add_argument('-pixel_perc_thr', type=float, default=30, help='upper limit for pixel percentage, default=30')
    parser.add_argument('-qc', type=int, default=1, help='set to 1 for qc output, default=1')
    parser.add_argument('-plot', type=int, default=0, help='set to 1 to show plots, default=0')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "matrix_removal")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    if args.qc == 1:
        qc_path = os.path.join(args.result_dir, 'quality_control')
        if not os.path.exists(qc_path):
            os.mkdir(qc_path)

    # read in data
    p = ImzMLParser(args.imzML_fl)
    df = utils.get_dataframe_from_imzML(args.imzML_fl, multi_index=False)
    sample_name = os.path.basename(args.imzML_fl).split('.')[0]
    # print(sample_name)
    
    matrix_cluster_file = [f for f in os.listdir(args.img_dir) if f.startswith(sample_name) and '_matrix_cluster' in f]
    sample_cluster_files = [f for f in os.listdir(args.img_dir) if f.startswith(sample_name) and 'cluster' in f and not 'cluster_0' in f]
    clusters = [f.split('_')[-1] for f in sample_cluster_files]
    # print(matrix_cluster_file)
    # print(sample_cluster_files)
    # print(clusters)

    # combine all cluster images to one --> tissue image
    tissue_img = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1), dtype=int)
    for fl in sample_cluster_files:
        img = tifffile.imread(os.path.join(args.img_dir, fl))
        tissue_img[img == 255] = 255
    tifffile.imwrite(os.path.join(qc_path, sample_name + '_tissue_clusters.tif'), tissue_img.astype('uint8'))

    matrix_img = tifffile.imread(os.path.join(args.img_dir, matrix_cluster_file[0]))
    tifffile.imwrite(os.path.join(qc_path, sample_name + '_matrix_cluster.tif'), matrix_img.astype('uint8'))
    matrix_spec = get_spectrum_from_bin_img(matrix_img, df)
    tissue_spec = get_spectrum_from_bin_img(tissue_img, df)

    # get correlation of each cluster to matrix and tissue
    corr_df = pd.DataFrame(index=clusters, columns=['matrix_pears', 'matrix_spear', 'tissue_pears', 'tissue_spear', 'pixel_perc'])
    for fl, cl in zip(sample_cluster_files, clusters):
        img = tifffile.imread(os.path.join(args.img_dir, fl))
        cl_spec = get_spectrum_from_bin_img(img, df)
        matrix_pears, _ = pearsonr(cl_spec, matrix_spec)
        tissue_pears, _ = pearsonr(cl_spec, tissue_spec)
        matrix_spear, _ = spearmanr(cl_spec, matrix_spec)
        tissue_spear, _ = spearmanr(cl_spec, tissue_spec)
        corr_df.at[cl, 'matrix_pears'] = matrix_pears
        corr_df.at[cl, 'tissue_pears'] = tissue_pears
        corr_df.at[cl, 'matrix_spear'] = matrix_spear
        corr_df.at[cl, 'tissue_spear'] = tissue_spear
        corr_df.at[cl, 'pixel_perc'] = (np.count_nonzero(img) / np.count_nonzero(tissue_img)) * 100
    # print(corr_df)
    corr_df.to_csv(os.path.join(qc_path, sample_name + '_corr.csv'), index=True)

    # visualize spearman correlation to matrix cluster
    fig, ax = plt.subplots()
    ax = corr_df.plot.bar(y='matrix_spear', rot=0)
    ax.axhline(y=args.matrix_corr_thr, color= 'red', linewidth=5,)
    plt.savefig(os.path.join(qc_path, sample_name + '_matrix_cluster_corr.svg'))
    plt.close()
    fig, ax = plt.subplots()
    ax = corr_df.plot.bar(y='pixel_perc', rot=0)
    ax.axhline(y=args.pixel_perc_thr, color= 'red', linewidth=5,)
    plt.savefig(os.path.join(qc_path, sample_name + '_cluster_pixel_perc.svg'))
    plt.close()

    # get classes with matrix corr > correlation threshold
    corr_df_filt = corr_df[(corr_df['matrix_spear'] > args.matrix_corr_thr) & (corr_df['pixel_perc'] < args.pixel_perc_thr)]
    matrix_corr_clusters = corr_df_filt.index
    # print(corr_df_filt)
    # print(matrix_corr_clusters)

    # combine matrix image with cluster images with matrix corr > correlation threshold
    extended_matrix_img = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1), dtype=int)
    for cl in matrix_corr_clusters:
        cl_img = tifffile.imread(os.path.join(args.img_dir, sample_cluster_files[clusters.index(cl)]))
        extended_matrix_img[cl_img == 255] = 255
    extended_matrix_img[matrix_img == 255] = 255
    tifffile.imwrite(os.path.join(args.result_dir, sample_name + '_extended_matrix_image.tif'), extended_matrix_img.astype('uint8'))

    # # get remaining clusters
    # corr_df_remain = corr_df[corr_df['matrix_spear'] <= args.matrix_corr_thr]
    # remain_clusters = corr_df_remain.index.to_numpy()
    #
    # # calculate correlation of remaining clusters to new tissue image
    # new_tissue_img = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1), dtype=int)
    # print(remain_clusters)
    # print(sample_cluster_files)
    # for cl in remain_clusters:
    #     print(sample_name + '_' + cl)
    #     id = [i for i in sample_cluster_files if sample_name + '_' + cl or sample_name + 'matrix_' + cl in i]
    #     if id:
    #         img = tifffile.imread(os.path.join(args.img_dir, id[0]))
    #         new_tissue_img[img == 255] = 255
    # tifffile.imwrite(os.path.join(qc_path, sample_name + '_new_tissue_clusters.tif'), new_tissue_img.astype('uint8'))
    #
    # extended_matrix_spec = get_spectrum_from_bin_img(extended_matrix_img, df)
    # new_tissue_spec = get_spectrum_from_bin_img(new_tissue_img, df)
    #
    # corr_df = pd.DataFrame(index=remain_clusters, columns=['matrix_pears', 'matrix_spear', 'tissue_pears', 'tissue_spear'])
    # print("remaining clusters = ", remain_clusters)
    # print("sample cluster files = ", sample_cluster_files)
    # for cl in remain_clusters:
    #     id = [i for i in sample_cluster_files if sample_name + '_' + cl or sample_name + 'matrix_' + cl in i]
    #     if id:
    #         img = tifffile.imread(os.path.join(args.img_dir, id[0]))
    #         cl_spec = get_spectrum_from_bin_img(img, df)
    #         matrix_pears, _ = pearsonr(cl_spec, matrix_spec)
    #         tissue_pears, _ = pearsonr(cl_spec, tissue_spec)
    #         matrix_spear, _ = spearmanr(cl_spec, matrix_spec)
    #         tissue_spear, _ = spearmanr(cl_spec, tissue_spec)
    #         corr_df.at[cl, 'matrix_pears'] = matrix_pears
    #         corr_df.at[cl, 'tissue_pears'] = tissue_pears
    #         corr_df.at[cl, 'matrix_spear'] = matrix_spear
    #         corr_df.at[cl, 'tissue_spear'] = tissue_spear
    # print(corr_df)
    # corr_df.to_csv(os.path.join(qc_path, sample_name + '_new_corr.csv'), index=True)

    # # get matrix pixels

    # bin_img_px_idx_np = np.nonzero(matrix_img)
    # print(bin_img_px_idx_np[0].shape)
    # print('no. of matrix pixels:', np.count_nonzero(matrix_img))
    # bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))
    # matrix_pixels_df = pd.DataFrame.from_dict({'x': bin_img_px_idx_np[1], 'y': bin_img_px_idx_np[0]})

    # # get matrix spectrum
    # matrix_df = pd.merge(left=df, right=matrix_pixels_df, on=['x', 'y'])
    # matrix_spec = matrix_df.iloc[:, 2:].mean(axis=0)
    # print(matrix_df)

    # # remove rows with matrix pixels from df
    # if args.pixel_removal == 1:
    #     print('no. pixels before matrix removal: ', df.shape[0])
    #     df_merge = pd.merge(df, matrix_df, how='outer', on=['x', 'y'], indicator=True)
    #     df = df.loc[df_merge['_merge'] == 'left_only']
    #     print('no. pixels after matrix removal: ', df.shape[0])

    # df_sub = df.iloc[:, 2:]
    # print(df_sub)
    # print(df)

    # # write matrix subtracted data
    # p = ImzMLParser(args.imzML_fl)
    # with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML_fl))) as writer:
    #     for i in tqdm(range(df_sub.shape[0])):
    #         writer.addSpectrum(df_sub.columns.to_numpy(), df_sub.iloc[i, :].to_numpy(),
    #                            (df.iloc[i, 0], df.iloc[i, 1], 0))

