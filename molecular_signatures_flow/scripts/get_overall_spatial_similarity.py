import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
from scipy.stats.mstats import pearsonr
import seaborn as sns
import itertools
from skimage.exposure import rescale_intensity
import matplotlib
import matplotlib.colors as colors

sys.path.append("..")
from pkg import utils
from pkg.plot import plot_venn2

pd.options.mode.chained_assignment = None  # default='warn'


def get_mz_img(pyx, msi_df, mz):
    coords = msi_df.index.tolist()
    msi_img = np.zeros(pyx).astype('uint16')
    for x_val, y_val in coords:
        msi_img[y_val , x_val] = msi_df.loc[(x_val, y_val), mz]
    return msi_img

# def get_mz_img(pyx, msi_df, mz, tol=0.00000001):
#     lower = mz - tol
#     upper = mz + tol
#     mzs_cols = msi_df.columns.to_numpy()
#     mzs_cols_tol_range = np.where((mzs_cols >= lower) & (mzs_cols <= upper))
#     msi_df = msi_df.iloc[:, mzs_cols_tol_range[0]]
#     msi_df['sum'] = msi_df.sum(axis=1)
#     coords = msi_df.index.tolist()
#     msi_img = np.zeros(pyx)
#     for x_val, y_val in coords:
#         msi_img[y_val, x_val] = msi_df.loc[(x_val, y_val), 'sum']
#     return msi_img


def get_combinations(lst):
    all_combinations = []
    for r in range(len(lst) + 1):
        combinations_object = itertools.combinations(lst, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list
    return all_combinations


def get_combi_mz_img(pyx, msi_df, mzs):
    coords = msi_df.index.tolist()
    msi_img = np.zeros(pyx).astype(np.uint16)
    msi_df_mzs = msi_df[list(mzs)]
    msi_df_mzs['mean'] = msi_df_mzs.mean(axis=1)
    for x_val, y_val in coords:
        msi_img[y_val, x_val] = msi_df_mzs.loc[(x_val, y_val), 'mean']
    return msi_img
    return msi_img


def get_pearson_corr(mz_combis, pyx, msi_df, img, contrast_stretch):
    mz_img = get_combi_mz_img(pyx, msi_df, mz_combis)
    mz_img = np.nan_to_num(mz_img)
    if contrast_stretch:
        p0, p99 = np.percentile(mz_img, (0, 99.9))
        mz_img = rescale_intensity(mz_img, in_range=(p0, p99))
        #mz_img = gaussian_filter(mz_img, sigma=1)
    mz_img = utils.NormalizeData(mz_img).ravel()
    pearson = pearsonr(x=img, y=mz_img)[0]
    return pearson


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_ranking(corr_df, n, out_file, label='Correlation', plot=False):
    """
    plot ranking with mean correlation
    :param corr_df: pd.DataFrame with m/z values as indices and samples and mean correlation as columns sorted
                    according to mean correlation
    :param n: number of m/z values to plot
    :param result_dir: directory to save plot
    :param plot: set to True for plotting
    """
    # plot ranking with top n m/z values
    fig, ax = plt.subplots()
    y_pos = np.arange(n)

    #cmap = matplotlib.cm.get_cmap('Reds_r')  # colorbar [0,1]
    cmap = matplotlib.colormaps.get_cmap('Reds_r')
    new_cmap = truncate_colormap(cmap, 0.2, 0.7)
    norm = matplotlib.colors.Normalize(vmin=min(y_pos), vmax=max(y_pos))
    colors = [matplotlib.colors.rgb2hex(new_cmap(norm(i))) for i in y_pos]
    ax.barh(y_pos, corr_df.head(n)['mean'].to_numpy(), align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.round(corr_df.head(n).index.to_numpy(), 2))
    #ax.set_yticklabels(np.round(corr_df.head(n)['m/z'].to_numpy(), 2))
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(label)
    plt.savefig(out_file)
    if plot:
        plt.show()
    plt.close()


def plot_violin(corr_df, n, out_file, label='Correlation', plot=False):
    """
    violin plot of correlation for top n m/z values
    :param corr_df: pd.DataFrame with m/z values as indices and samples and mean correlation as columns sorted
                    according to mean correlation
    :param n: number of m/z values to plot
    :param result_dir: directory to save plot
    :param plot: set to True for plotting
    """
    plt.figure(figsize=(10, 5))
    top_corr_df = corr_df.head(n).iloc[:, :-1]
    stacked = top_corr_df.stack().reset_index()
    stacked = stacked.rename(columns={'level_0': 'm/z values', 'level_1': 'Samples', 0: label})
    stacked = stacked.round({"m/z values": 4})
    sns.violinplot(x="m/z values", y=label, data=stacked, inner=None)
    sns.swarmplot(x="m/z values", y=label, data=stacked, color="white", edgecolor="gray")
    plt.savefig(out_file)
    if plot:
        plt.show()
    plt.close()


def merge_files(base_df, file_dir, file_list):
    for fl in file_list:
        fl_df = pd.read_csv(os.path.join(file_dir, fl), index_col=0)
        fl_df.set_index(fl_df.index.to_numpy().astype('float32'), inplace=True)
        base_df = pd.merge(base_df, fl_df, left_index=True, right_index=True)
    return base_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate spatial correlation of all m/z values of multiple imzML'
                                                 'files')
    parser.add_argument('dir', type=str, help='directory with correlation files')
    # parser.add_argument('imzML_dir', type=str, help='directory with imzML files')
    # parser.add_argument('img_dir', type=str, help='directory with image files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store results')
    parser.add_argument('-n', type=int, default=10, help='select best combination out of top n m/z values')
    # parser.add_argument('-contrast_stretch', type=bool, default=False, help='set to True for contrast stretching')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(args.dir, 'overall')
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)

    # read in data
    # imzML_files = [file for file in os.listdir(args.imzML_dir) if file.endswith('.imzML')]
    # img_files = [file for file in os.listdir(args.img_dir) if file.endswith('.tif') and 'pos' in file]
    # imzML_files = np.array(sorted(imzML_files, key=lambda x: int(x.split('.')[0][-2:])))
    # img_files = np.array(sorted(img_files, key=lambda x: int(x.split('.')[0][-2:])))
    pearson_files = [file for file in os.listdir(args.dir) if file.startswith('pearson') and file.endswith('.csv')]
    cosine_files = [file for file in os.listdir(args.dir) if file.startswith('cosine') and file.endswith('.csv')]

    # print(imzML_files)
    # print(img_files)
    # print(pearson_files)
    # print(cosine_files)

    df = pd.read_csv(os.path.join(args.dir, pearson_files[0]))
    #df = pd.read_csv(os.path.join(args.dir, pearson_files[0])).iloc[:10, :]
    mzs = df.iloc[:, 0].to_numpy().astype(np.float64)

    pearson_df = pd.DataFrame(index=mzs)
    cosine_df = pd.DataFrame(index=mzs)
    pearson_df.set_index(pearson_df.index.to_numpy().astype('float32'), inplace=True)
    cosine_df.set_index(cosine_df.index.to_numpy().astype('float32'), inplace=True)
    pearson_df = merge_files(pearson_df, args.dir, pearson_files)
    cosine_df = merge_files(cosine_df, args.dir, cosine_files)

    # get mean correlation of all samples and sort accordingly
    pearson_df['mean'] = pearson_df.mean(axis=1)
    pearson_df = pearson_df.sort_values(by='mean', ascending=False)
    cosine_df['mean'] = cosine_df.mean(axis=1)
    cosine_df = cosine_df.sort_values(by='mean', ascending=False)
    # drop 1.0 values in cosine similarity which was caused due to error
    for col in cosine_df.columns:
        cosine_df = cosine_df[cosine_df[col] != 1.0]

    # plot ranking and violin plot
    plot_ranking(pearson_df, args.n, os.path.join(args.result_dir, 'barplot_pearson_corr.svg'),
                 'Pearson Correlation',
                 args.plot)
    plot_violin(pearson_df, args.n, os.path.join(args.result_dir, 'violinplot_pearson_corr.svg'),
                'Pearson Correlation',
                args.plot)
    plot_ranking(cosine_df, args.n, os.path.join(args.result_dir, 'barplot_cosine_sim.svg'), 'Cosine Similarity',
                 args.plot)
    plot_violin(cosine_df, args.n, os.path.join(args.result_dir, 'violinplot_cosine_sim.svg'), 'Cosine Similarity',
                args.plot)

    # save top m/z images of all samples

    # get intersection of similarity measures
    top_n_mz_pearson = set(pearson_df.head(args.n).index.to_list())
    top_n_mz_cosine = set(cosine_df.head(args.n).index.to_list())
    plot_venn2(label1='Pearson Corr.', label2='Cosine Sim.', data1=top_n_mz_pearson, data2=top_n_mz_cosine,
               title='Overlay of similarity measures',
               output_file=os.path.join(args.result_dir, 'venn_similarity_measures.svg'),
               plot=args.plot)
    # top_mz_intersec = list(top_n_mz_pearson & top_n_mz_cosine)
    top_mz_intersec = top_n_mz_pearson

    # save spatial correlation as csv file
    pearson_df.index.rename('m/z', inplace=True)
    cosine_df.index.rename('m/z', inplace=True)

    pearson_df.to_csv(os.path.join(args.result_dir, 'overall_spatial_ranking_pearson.csv'))
    cosine_df.to_csv(os.path.join(args.result_dir, 'overall_spatial_ranking_cosine.csv'))

    # print(pearson_df)
    # print(cosine_df)

    # # create directory for top m/z and save image for each imzml file
    # print('saving top m/z images...')
    # for i, mz in enumerate(tqdm(top_mz_intersec)):
    #     mz_dir = os.path.join(args.result_dir, str(round(mz, 4)).replace('.', '_') + 'mz')
    #     if not os.path.exists(mz_dir):
    #         os.mkdir(mz_dir)
    #     for imzML_fl in imzML_files:
    #         sample_num = imzML_fl.split('.')[0][-2:]
    #         p = ImzMLParser(os.path.join(args.imzML_dir, imzML_fl))
    #         pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
    #         msi_df = utils.get_dataframe_from_imzML(os.path.join(args.imzML_dir, imzML_fl), multi_index=True)
    #         top_mz_img = get_mz_img(pyx, msi_df, mz)
    #         if args.contrast_stretch:
    #             p0, p99 = np.percentile(top_mz_img, (0, 99.9))
    #             top_mz_img = rescale_intensity(top_mz_img, in_range=(p0, p99))
    #         top_mz_img = (utils.NormalizeData(top_mz_img) * 255).astype('uint8')
    #         tifffile.imwrite(os.path.join(mz_dir, sample_num + '.tif'), data=top_mz_img)

    # # test spatial correlation of combination of top m/z values
    # combis = get_combinations(top_mz_intersec)
    #
    # # test combination of m/z values
    # print("calculating correlation of combination of top correlating m/z values to reference image...")
    # combi_corr_df = pd.DataFrame(index=combis[1:])
    # print(combi_corr_df)
    # for imzML_fl, img_fl in zip(imzML_files, img_files):
    #     print("processing file {}".format(imzML_fl))
    #     sample_num = imzML_fl.split('.')[0][-2:]
    #     img = utils.NormalizeData(tifffile.imread(os.path.join(args.img_dir, img_fl))).ravel()
    #     p = ImzMLParser(os.path.join(args.imzML_dir, imzML_fl))
    #     pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
    #     msi_df = utils.get_dataframe_from_imzML(os.path.join(args.imzML_dir, imzML_fl), multi_index=True)  # data frame of imzMl file containing all pixels
    #     # get correlation for each m/z combination
    #     pearson = []
    #
    #     # create a process pool that uses all cpus
    #     with multiprocessing.Pool() as pool:
    #         # call the function for each item in parallel
    #         for result in pool.map(partial(get_pearson_corr, pyx=pyx, msi_df=msi_df, img=img,
    #                                        contrast_stretch=args.contrast_stretch), combis[1:]):
    #             pearson.append(result)
    #     combi_corr_df[sample_num] = pearson
    #
    # # get mean correlation of all samples and sort accordingly
    # combi_corr_df['mean'] = combi_corr_df.mean(axis=1)
    # combi_corr_df = combi_corr_df.sort_values(by='mean', ascending=False)
    #
    # # save spatial correlation as csv file
    # combi_corr_df.to_csv(os.path.join(args.result_dir, 'combi_top_spatial_ranking.csv'))
    #
    # print(combi_corr_df)
    #
    # # save image with top combination of m/z values
    # top_mz_combi = combi_corr_df.index.tolist()[0]
    # print("Top m/z combination=", top_mz_combi)
    # for imzML_fl, img_fl in zip(imzML_files, img_files):
    #     sample_num = imzML_fl.split('.')[0]
    #     p = ImzMLParser(os.path.join(args.imzML_dir, imzML_fl))
    #     pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
    #     msi_df = utils.get_dataframe_from_imzML(os.path.join(args.imzML_dir, imzML_fl),
    #                                             multi_index=True)  # data frame of imzMl file containing all pixels
    #     top_mz_combi_img = get_combi_mz_img(pyx, msi_df, top_mz_combi)
    #     if args.contrast_stretch:
    #         p0, p99 = np.percentile(top_mz_combi_img, (0, 99.9))
    #         top_mz_combi_img = rescale_intensity(top_mz_combi_img, in_range=(p0, p99))
    #     top_mz_combi_img = (utils.NormalizeData(top_mz_combi_img) * 255).astype('uint8')
    #     tifffile.imwrite(os.path.join(args.result_dir, 'top_mz_combi_' + sample_num + '.tif'), data=top_mz_combi_img)
    #
    # # save ranking with top combi
    # print("pearson_df=", pearson_df)
    # print("combi_corr_df=", combi_corr_df)
    # highest_corr = combi_corr_df.iloc[0, -1]
    # print("highest_corr=", highest_corr)
    # line = pd.DataFrame({"mean": highest_corr}, index=[0])
    # pearson_df = pearson_df.append(line, ignore_index=False)
    # print("pearson_df after adding combi corr=", pearson_df)
    # pearson_df = pearson_df[pearson_df.columns[-1]].to_frame()
    # pearson_df = pearson_df.sort_values(by='mean', ascending=False)
    # print(pearson_df)
    #
    # plot_ranking(pearson_df, args.n+1, os.path.join(args.result_dir, 'barplot_pearson_corr_combi.svg'), 'Pearson Correlation',
    #              args.plot)

    # df = pd.DataFrame.from_dict({'m/z': [0, 742.5673686783217, 720.5883800317306, 744.587275733194, 768.586171434659, 766.571264149723, 746.6021830181304, 709.5138896194924, 537.5218037589977, 758.5366338596712, 683.4950868631547],
    #                              'mean': [0.3934710223693078, 0.3633994812179265, 0.35743498702639426, 0.34609191687121216, 0.3425370690824923, 0.34161627829517865, 0.34061549118175166, 0.3332593579770613, 0.31089093970821624, 0.30738022701632073, 0.3041022574030876]})
    #
    # plot_ranking(df, 11, '/home/phispa/UPEC/MSI/3_bladders/peakpicking/alignment/deisotoping/intranorm_median/internorm_median/spatial_coherence/leadmasses1/overall/overall_spatial_ranking_pearson_combi.svg', plot=True)
    #
