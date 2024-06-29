import numpy as np
import sklearn.datasets
import sklearn.neighbors
import umap
import matplotlib.pyplot as plt
import sys
import argparse
import os
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import pickle
import seaborn as sns
import pandas as pd
import shutil
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils
from pkg import plot as plot_functions



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts smample outliers based on UMAP')
    parser.add_argument('umap_cluster_file', type=str, help='csv file of UMAP embedding and clusters')
    parser.add_argument('imzML_dir', type=str, help='directory to imzML files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to save results')
    parser.add_argument('-cluster_thr', type=int, default=80,
                        help='cluster pixel percentage which must be covered by one sample to be considered a sample-specific cluster (SSC), default=80')
    parser.add_argument('-sample_thr', type=float, default=50,
                        help='sample pixel percentage which must be covered by sample-specific cluster pixels to be considered a outlier sample, default=50')
    parser.add_argument('-remove_sample_outliers', type=int, default=0, help='set to 1 to remove sample outliers, default=0')
    parser.add_argument('-remove_ssc', type=lambda x: utils.booltoint(x), default=1, help='set to True to remove SSC, default=1')
    args = parser.parse_args()

    imzML_files = [f.split('.')[0] for f in os.listdir(args.imzML_dir) if f.endswith('.imzML')]

    # create directories to save results
    if args.result_dir == '':
        args.result_dir = os.path.join(args.imzML_dir, "outlier_removal")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    qc_dir = os.path.join(args.result_dir, 'quality_control')
    if not os.path.exists(qc_dir):
        os.mkdir(qc_dir)

    df = pd.read_csv(args.umap_cluster_file)
    clusters = np.unique(df['label'].to_numpy())

    # samples_covering_single_clusters contains sample which cover more than cluster_thr pixels for cluster defined in clusters_with_samples_covering_most_pixels
    cluster_thr = args.cluster_thr
    samples_covering_single_clusters = []               
    clusters_with_samples_covering_most_pixels = []
    sample_px_perc = []

    # create dict with sample colors
    samples = np.unique(df['sample'].to_numpy())
    # print(samples)
    sample_colours = {}
    for i, smpl in enumerate(samples):
        sample_colours[smpl] = 'C' + str(i)
    # print(sample_colours)

    for cl in clusters:
        cl_df = df[df['label'] == cl]
        cl_pixels = cl_df.shape[0]
        sample_count = ((cl_df.groupby('sample')[['label']].count()).div(cl_pixels)).mul(100)
        samples_with_pixels_above_thr = (sample_count.index.to_numpy()[sample_count['label'].to_numpy() > cluster_thr]).tolist()
        samples_covering_single_clusters.extend(samples_with_pixels_above_thr)

        if not np.all((sample_count['label'].to_numpy() > cluster_thr) == False):
            clusters_with_samples_covering_most_pixels.append(cl)

        # pie chart with sample pixels
        fig, ax = plt.subplots()
        lbls = sample_count.index.to_numpy()
        ax.pie(sample_count['label'].to_numpy(), labels=lbls, autopct='%1.1f%%',
               colors=[sample_colours[key] for key in lbls])
        plt.savefig(os.path.join(qc_dir, str(cl) + '_sample_pixels.svg'))
        plt.close()

    for cl, smpl in zip(clusters_with_samples_covering_most_pixels, samples_covering_single_clusters):
        # get numbe of sample pixels
        smpl_pixels = df[df['sample'] == smpl].shape[0]
        smpl_pixels_in_cl = df[(df['sample'] == smpl) & (df['label'] == cl)].shape[0]
        smpl_cl_pix_perc = (smpl_pixels_in_cl / smpl_pixels) * 100
        sample_px_perc.append(smpl_cl_pix_perc)
    
    # print(samples_covering_single_clusters)
    # print(clusters_with_samples_covering_most_pixels)
    # print(sample_px_perc)
    df_result = pd.DataFrame.from_dict({'cluster': clusters_with_samples_covering_most_pixels, 'sample': samples_covering_single_clusters, 'sample_pixel_perc': sample_px_perc})
    df_px_perc_sum = df_result.groupby('sample')['sample_pixel_perc'].sum()
    df_result.to_csv(os.path.join(qc_dir, 'sample_pixel_perc.csv'))

    # barplot
    df_result_pivot = df_result.pivot(index='sample', columns='cluster')
    df_result_pivot.columns = df_result_pivot.columns.get_level_values(1)
    df_result_pivot.plot(kind='bar', stacked=True, figsize=(8, 8))
    plt.xticks(rotation=45)
    plt.axhline(y=args.sample_thr, color = 'k', linestyle = '--')
    plt.legend(loc='upper right')
    plt.ylabel('sample pixels (%)')
    plt.savefig(os.path.join(qc_dir, 'barplot.svg'))

    # only save non-outlier samples
    # print(df_px_perc_sum)
    # print(df_result)
    sample_outliers = df_px_perc_sum[df_px_perc_sum > args.sample_thr].index.to_list()
    print('identified sample outliers:', sample_outliers)

    if args.remove_sample_outliers == 1:
        imzML_files_without_outliers = set(imzML_files) - set(sample_outliers)
    else:
        imzML_files_without_outliers = imzML_files
    for f in imzML_files_without_outliers:
        shutil.copy(os.path.join(args.imzML_dir, f + '.imzML'), os.path.join(args.result_dir, f + '.imzML'))
        shutil.copy(os.path.join(args.imzML_dir, f + '.ibd'), os.path.join(args.result_dir, f + '.ibd'))

    # remove SSC pixels from datasets
    if args.remove_ssc:
        # remove outlier samples from result file
        df_result = df_result[~df_result['sample'].isin(sample_outliers)]
        # print(df_result)

        # create dict with sample: [cluster_1, ... , cluster_n]
        sample_outlier_cluster_dict = df_result.groupby('sample')['cluster'].apply(list).to_dict()
        # print(sample_outlier_cluster_dict)

        # remove cluster pixels from each imzML file containing outlier clusters
        for sample in sample_outlier_cluster_dict:
            # get dataframe of sample and clusters
            cluster_fl_df = df[df['sample'] == sample]
            cluster_fl_df = cluster_fl_df[cluster_fl_df['label'].isin(sample_outlier_cluster_dict[sample])]
            # print(cluster_fl_df)
            cluster_fl_df = cluster_fl_df[['x', 'y']]
            # print(cluster_fl_df)

            # remove cluster pixels from fl_df
            fl_df = utils.get_dataframe_from_imzML(os.path.join(args.imzML_dir, sample + '.imzML'), multi_index=False)
            df_merge = pd.merge(fl_df, cluster_fl_df, how='outer', on=['x', 'y'], indicator=True)
            pixel_removed_df = fl_df.loc[df_merge['_merge'] == 'left_only']
            # print(pixel_removed_df)
            print('no. pixels before pixel removal: ', fl_df.shape[0])
            print('no. pixels after pixel removal: ', pixel_removed_df.shape[0])

            # write pixel removed data
            pixel_removed_spec_df = pixel_removed_df.iloc[:, 2:]

            with ImzMLWriter(os.path.join(args.result_dir, sample + '.imzML')) as writer:
                for i in tqdm(range(pixel_removed_df.shape[0])):
                    writer.addSpectrum(pixel_removed_spec_df.columns.to_numpy(),
                                       pixel_removed_spec_df.iloc[i, :].to_numpy(),
                                       (pixel_removed_df.iloc[i, 0], pixel_removed_df.iloc[i, 1], 0))


    

