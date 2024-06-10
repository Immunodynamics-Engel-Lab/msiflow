import argparse
import pandas as pd
import os
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import warnings
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.append("..")
from pkg import utils

warnings.filterwarnings('ignore', module='pyimzml')


def plot_spectrum(mzs, intensities, plot=False, output_file=''):
    plt.plot(mzs, intensities)
    plt.xlabel('m/z [Da]')
    plt.ylabel('Intensities [a.u.]')
    if output_file != '':
        plt.savefig(output_file)
    if plot:
        plt.show()
    plt.close()


def plot_embedding(df, col, output_dir, method, pca=None, plot=False):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=df['dim 1'], y=df['dim 2'], s=70, hue=df[col])

    if method == 'pca' and pca:
        plt.xlabel('First principal component ({}%)'.format(np.round(pca.explained_variance_ratio_[0] * 100, 2)))
        plt.ylabel('Second principal component ({}%)'.format(np.round(pca.explained_variance_ratio_[1] * 100, 2)))
    elif method == 't-sne':
        plt.xlabel('t-sne 1')
        plt.ylabel('t-sne 2')
    else:
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
    plt.savefig(os.path.join(output_dir, '2D_{}_colored_by_{}.pdf'.format(method, col)))
    if plot:
        plt.show()
    plt.close()


def get_summarized_spectra(result_dir, imzML_dir='', sum_file='', method='mean', plot=False, mz_rows=False, save_plots=False):
    if sum_file != '':
        df_sum_all = pd.read_csv(sum_file, index_col=0)
    else:
        # get all imzML files
        imzML_files = [f for f in os.listdir(imzML_dir) if os.path.isfile(os.path.join(imzML_dir, f))
                    and f.endswith('.imzML')]
        print('found {} sample files in {}'.format(len(imzML_files), os.path.basename(imzML_dir)))

        # len_mzs = []
        # set_mzs = set()
        # for file in imzML_files:
        #     _, _, mzs = utils.get_spectra_coords_arrays(os.path.join(imzML_dir, file))
        #     len_mzs.append(len(mzs.tolist()))
        #     set_mzs.update(set(mzs.tolist()))
        # if all(x == len_mzs[0] for x in len_mzs):
        #     print("consistent m/z vector of {} values".format(len_mzs[0]))
        # else:
        #     print("no consistent m/z vector")
        _, _, mzs = utils.get_spectra_coords_arrays(os.path.join(imzML_dir, imzML_files[0]))

        # create dataframe for all summarized spectra
        # df_sum_all = pd.DataFrame(index=list(set_mzs))
        df_sum_all = pd.DataFrame(index=mzs)

        print('extracting summarized spectra ...')
        for file in tqdm(imzML_files):
            # create pandas.DataFrame from imzML
            df = utils.get_dataframe_from_imzML(os.path.join(imzML_dir, file), multi_index=True)
            if method == 'mean':
                df_sum = df.mean(axis=0, skipna=True).T.to_frame()
            else:
                df_sum = df.median(axis=0, skipna=True).T.to_frame()
            df_sum = df_sum.rename(columns={0: file.split('.')[0]})
            df_sum_all = df_sum_all.merge(df_sum, how='outer', left_index=True, right_index=True)

    if mz_rows:
        df_sum_all_final = df_sum_all
    else:
        df_sum_all_final = df_sum_all.T
    #print(df_sum_all_final)

    if save_plots:
        qc_dir = os.path.join(args.result_dir, 'quality_control')
        if not os.path.exists(qc_dir):
            os.mkdir(qc_dir)

        # plot spectra
        for index, row in df_sum_all.T.iterrows():
            plt.plot(df_sum_all.T.columns, row, label=index)
        plt.legend()
        plt.xlabel('m/z')
        plt.ylabel('Intensities [a.u.]')
        plt.savefig(os.path.join(qc_dir, 'summarized_spectra.svg'))
        plt.close()

        # plot boxplot with summarized m/z intensities
        #df_sum_all = df_sum_all.applymap(math.log10)
        # flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='green', alpha=0.5)
        # sns.boxplot(data=df_sum_all, flierprops=flierprops)
        df_sum_all = df_sum_all[sorted(df_sum_all.columns)]
        ax = sns.boxplot(data=df_sum_all, showfliers=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.xlabel("samples", size=12)
        plt.ylabel("intensities", size=12)
        plt.savefig(os.path.join(qc_dir, 'samples_boxplot.svg'))
        plt.close()

        # sns.boxplot(data=df_sum_all.applymap(math.log10), flierprops=flierprops)
        if method !='median':
            ax = sns.boxplot(data=df_sum_all.applymap(math.log10), showfliers=False)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.xlabel("samples", size=12)
            plt.ylabel("log(intensities)", size=12)
            plt.savefig(os.path.join(qc_dir, 'samples_boxplot_log.svg'))
            plt.close()

        if len(imzML_files) > 1:
            # plot correlation matrix
            df_sum_all_final.sort_index(axis=1, inplace=True)
            corr = df_sum_all_final.corr()
            g = sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
            #g.ax_heatmap.set_aspect('equal')
            plt.savefig(os.path.join(qc_dir, 'corr_heatmap.svg'))
            plt.close()

            # plot PCA
            # standardize data
            if mz_rows:
                samples = df_sum_all.columns.to_numpy()
                df_sum_all = df_sum_all.T
            intensities = df_sum_all.to_numpy()

            scaler = StandardScaler()
            scaler.fit(intensities)
            intensities_scaled = scaler.transform(intensities)
            intensities_scaled = np.nan_to_num(intensities_scaled)
            # plot number of possible components and the explained variance
            pca = PCA(n_components=min(intensities_scaled.shape))
            pca.fit(intensities_scaled)
            #print('Variance explained by components = {}'.format(pca.explained_variance_ratio_ * 100))
            variance = np.cumsum(pca.explained_variance_ratio_ * 100)
            plt.plot(variance)
            plt.xlabel('Number of components')
            plt.ylabel('Explained variance (%)')
            plt.savefig(os.path.join(qc_dir, 'Explained_variance_pca.pdf'))
            if plot:
                plt.show()
            plt.close()
            # 2D PCA
            pca_2 = PCA(n_components=2)
            embedding = pca_2.fit_transform(intensities_scaled)

            embedding_df = pd.DataFrame(data=embedding, columns=['dim 1', 'dim 2'])
            final_df = pd.concat((embedding_df, pd.DataFrame.from_dict({'sample': samples})), axis=1)
            # final_df = pd.concat([embedding_df, df_data[['group']]], axis=1)
            #print(final_df)
            plot_embedding(df=final_df, col='sample', output_dir=qc_dir, method='pca', pca=pca_2, plot=plot)

    # save summarized spectra
    #print(df_sum_all_final)
    df_sum_all_final.to_csv(os.path.join(result_dir, 'summarized.csv'), index=True)

    #     spectra, coords, mzs = utils.get_spectra_coords_arrays(os.path.join(imzML_dir, file))
    #     df_spectra = pd.DataFrame(spectra, columns=mzs)
    #     df_spectra.insert(loc=0, column='x', value=list(coords[:, 0]))
    #     df_spectra.insert(loc=1, column='y', value=list(coords[:, 1]))
    #
    #     # extract summarized spectrum
    #     df_sum_spectra = utils.get_summarized_spectrum(df_spectra, method=method)
    #     summarized_spectrum = df_sum_spectra.to_numpy().ravel()
    #     if save_plots:
    #         plot_spectrum(mzs=mzs, intensities=summarized_spectrum,
    #                       output_file=os.path.join(result_dir, file.split('.')[0] + '_summarized_spectrum.pdf'),
    #                       plot=plot)
    #
    #     # insert summarized spectrum to data list
    #     row = list(summarized_spectrum)
    #     row.insert(0, file.split('.')[0])
    #     data.append(row)
    #
    # # save summarized spectra as csv
    # df_sum_all = pd.DataFrame(data, columns=cols)
    #
    # if mz_rows:
    #     spectra = df_sum_all.iloc[:, 1:].to_numpy()
    #     mzs = df_sum_all.columns[1:].to_numpy().astype(float)
    #     idx = df_sum_all['ID'].to_list()
    #     df_sum_all = pd.DataFrame(columns=mzs, data=spectra, index=idx)
    #     df_sum_all = df_sum_all.transpose()
    #
    #df_sum_all.to_csv(os.path.join(result_dir, 'summarized.csv'), index=mz_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates summarized spectrum of a MSI data set saved as imzML file')
    parser.add_argument('imzML_dir', type=str, help='directory containing imzML files')
    parser.add_argument('-sum_file', type=str, default='', help='path to summarized file if already available')
    parser.add_argument('-method', type=str, default='mean', help='method to summarize')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store results')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot summarized spectrum')
    parser.add_argument('-mz_rows', type=bool, default=True, help='set to True so that m/z values are rows')
    parser.add_argument('-qc', type=bool, default=True, help='set to True for quality control output')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(args.imzML_dir, 'summarized_' + args.method)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    get_summarized_spectra(imzML_dir=args.imzML_dir, sum_file=args.sum_file, method=args.method, result_dir=args.result_dir, plot=args.plot,
                           mz_rows=args.mz_rows, save_plots=args.qc)

