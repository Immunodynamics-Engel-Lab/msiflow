import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import random
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm import tqdm
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils
from pkg.plot import plot_img_heatmap


def get_scfactors(spectra, method='median'):
    if method == 'mfc':
        median_spectrum = np.nanmedian(spectra, axis=0)     # median sample spectrum
        ratios = np.divide(spectra, median_spectrum)        # ratios between intensities and corr. reference intenisties
        scfactors = np.nanmedian(ratios, axis=1)            # median of ratios
    elif method == 'mean':
        scfactors = np.nanmean(spectra, axis=1)
    elif method == 'sum':
        scfactors = np.nansum(spectra, axis=1)
    elif method == 'median':
        scfactors = np.nanmedian(spectra, axis=1)
    return scfactors


def get_sc_img(pyx, df):
    coords = df.index.tolist()
    sc_img = np.zeros(pyx).astype(np.uint8)
    for x_val, y_val in coords:
        sc_img[y_val - 1, x_val - 1] = df.loc[(x_val, y_val), 'scfactor']
    return sc_img


if __name__ == '__main__':
    # spec = np.array([[1, 5, 2],
    #                  [2, 6, 3],
    #                  [2, 5, 1],
    #                  [3, 8, 2],
    #                  [3, 7, 3]])
    # df = pd.DataFrame(columns=['mz1', 'mz2', 'mz3'], data=spec)
    # sc_facs = get_scfactors(spec, method='mfc')
    # df_norm = df.divide(sc_facs, axis='rows')
    # print(sc_facs)
    # print(df_norm)
    parser = argparse.ArgumentParser(description='Performs intranormalization of MSI data')
    parser.add_argument('imzML_fl', type=str, help='imzML file')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result')
    parser.add_argument('-method', type=str, default='median', help='method for normalization')
    parser.add_argument('-qc', type=int, default=1, help='set to 1 for quality control output')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "intranormed")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    fl_name = os.path.basename(args.imzML_fl).split('.')[0]
    p = ImzMLParser(args.imzML_fl)
    df = utils.get_dataframe_from_imzML(args.imzML_fl, multi_index=True)
    # print(df)

    # ignore mz features and spectra with all zeros or nans
    df = df.replace(0, np.nan)
    df = df.dropna(how='all', axis='columns')
    # df = df.dropna(how='all', axis=0)
    # df = df.dropna(how='all', axis=1)
    # print(df)

    # get scaling factors for spectra
    spec = df.to_numpy()
    scfacs = get_scfactors(spec, args.method)

    # set divisor to 1 if scaling factor is 0 to prevent devide by 0 error
    scfacs[np.isnan(scfacs)] = 1
    scfacs[scfacs == 0] = 1

    # apply scaling factor
    df = df.replace(np.nan, 0)
    df_norm = df.divide(scfacs, axis='rows')
    # print(df_norm)

    # quality control
    if args.qc == 1:
        qc_dir = os.path.join(args.result_dir, 'quality_control')
        if not os.path.exists(qc_dir):
            os.mkdir(qc_dir)

        # plot image with pixelwise scaling factors
        df_scfacs = pd.DataFrame(index=df_norm.index, columns=['scfactor'], data=scfacs)
        pyx = (p.imzmldict["max count of pixels y"], p.imzmldict["max count of pixels x"])
        scfacs_img = get_sc_img(pyx, df_scfacs)
        plot_img_heatmap(scfacs_img, output_file=os.path.join(qc_dir, fl_name + '_scfactors.svg'), plot=False,
                         cmap='jet', ticks=True)
        
        # plot boxplot of randomly selected pixels
        random_list = []
        for i in range(5):
            r = random.randint(0, df.shape[0])
            if r not in random_list:
                random_list.append(r)
        # print("random list=", random_list)

        # get dataframe of random pixels
        df_rand = df.iloc[random_list]
        df_norm_rand = df_norm.iloc[random_list]
        rand_pixels = df_rand.index.to_numpy()

        # plot pixel spectrum
        for index, row in df_rand.iterrows():
            plt.plot(df_rand.columns, row, label=index)
            plt.legend()
            plt.ylim([0, df_rand.to_numpy().max()])
            plt.xlabel('m/z')
            plt.ylabel('Intensities [a.u.]')
            plt.savefig(os.path.join(qc_dir, fl_name + '_' + str(index) + '_before_norm.svg'))
            plt.close()
        for index, row in df_norm_rand.iterrows():
            plt.plot(df_norm_rand.columns, row, label=index)
            plt.legend()
            plt.ylim([0, df_norm_rand.to_numpy().max()])
            plt.xlabel('m/z')
            plt.ylabel('Intensities [a.u.]')
            plt.savefig(os.path.join(qc_dir, fl_name + '_' + str(index) + '_after_norm.svg'))
            plt.close()

        # plot boxplots
        flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='green', alpha=0.5)

        ax = sns.boxplot(data=df_rand.T, showfliers=True, showmeans=True, flierprops=flierprops)
        plt.ylabel("intensities", size=12)
        plt.savefig(os.path.join(qc_dir, fl_name + '_boxplot_before_norm.svg'))
        plt.close()

        ax = sns.boxplot(data=df_norm_rand.T, showfliers=True, showmeans=True, flierprops=flierprops)
        plt.ylabel("intensities", size=12)
        plt.savefig(os.path.join(qc_dir, fl_name + '_boxplot_after_norm.svg'))
        plt.close()

    # save imzML with normalised data
    with ImzMLWriter(os.path.join(args.result_dir, fl_name + '.imzML')) as writer:
        for index, row in df_norm.iterrows():
            writer.addSpectrum(df_norm.columns.to_numpy(), row.to_numpy(), (index[0], index[1], 0))

