import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots heatmap')
    parser.add_argument('input', type=str, help='csv file with m/z and intensities')
    parser.add_argument('output', type=str, help='output file with heatmap')
    parser.add_argument('-cmap', type=str, default='bwr', help='colormap')
    parser.add_argument('-row_norm', type=lambda x: utils.booltoint(x), default=True,
                        help='True=rowwise normalisation, False=no normalisation')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    df_ints = pd.read_csv(args.input, index_col=0, delimiter=',')

    if not df_ints.empty:
        cols = df_ints.columns.tolist()
        cols = [i for i in cols if not 'Fold change' in i and not 'p-value' in i and not 'SNR' in i
                and 'Unnamed: 0' not in i and not 'Lipid class' in i and not 'Lipid class no.' in i]
        df_ints = df_ints[cols]

        if 'Name' in df_ints:
            df_ints.set_index('Name', drop=True, inplace=True)
            df_ints.index.name = 'Name'
        else:
            df_ints.index = np.round(df_ints.index.to_numpy(), 2)
            df_ints.index.name = 'm/z'

        df_ints.sort_index(axis=1, inplace=True)
        # print(df_ints)
        # min-max scale
        # scaler = MinMaxScaler()
        # df_intervals_ints = pd.DataFrame(scaler.fit_transform(df_intervals_ints.T).T, columns=df_intervals_ints.columns,
        #                                 index=idx)
        # print(df_intervals_ints)

        #fig, ax = plt.subplots(figsize=(4, 10))
        #kws = dict(cbar_kws=dict(location='bottom'), figsize=(6, 6))
        if args.row_norm == 1:
            g = sns.clustermap(df_ints, col_cluster=True, row_cluster=True, cmap=args.cmap, standard_scale=0, square=True,
                               metric='euclidean', method='complete', yticklabels=1, xticklabels=1)
        else:
            g = sns.clustermap(df_ints, col_cluster=True, row_cluster=True, cmap=args.cmap, standard_scale=None,
                               metric='euclidean', method='complete', yticklabels=1, xticklabels=1)
        g.ax_heatmap.set_aspect('equal')
        #g.ax_row_dendrogram.set_visible(True)
        # g.ax_cbar.set_visible(False)
        g.ax_heatmap.tick_params(labelsize=6)
    plt.savefig(args.output)

    if args.plot:
        plt.show()







    # df_ints = pd.read_csv(args.intensities_file, index_col=0, delimiter=',')
    # df_prots = pd.read_csv(args.proteins_file, delimiter=',')
    #
    # print(df_ints)
    # print(df_prots)
    #
    # # # filter specific accessions
    # # prot_list = ['P20152', 'P35441', 'Q61362', 'P14824', 'P42227', 'Q9D659', 'Q6P9R2', 'Q7TMR0', 'Q8BVF2', 'Q8BTI9',
    # #              'Q7TPH6',
    # #              'B9EJ86', 'Q69ZK0', 'O09126', 'Q501J7', 'Q05793', 'Q61292', 'Q60675', 'Q9WTR5']
    # # prot_list = ['Q501J7', 'P14824', 'Q05793', 'Q8BVF2', 'Q7TPH6', 'Q61292', 'Q69ZK0', 'Q8BTI9', 'P35441', 'Q9D659',
    # #              'B9EJ86', 'O09126', 'Q7TMR0', 'Q60675', 'P42227', 'Q6P9R2', 'Q9WTR5', 'P20152']
    # # df_prots = df_prots[df_prots['Accession'].isin(prot_list)]
    #
    # # get gene names from description
    # df_ints['Gene name'] = df_ints['Description'].str.extract('GN=([a0-z9]{3,15}) ')
    #
    # # filter intensities file down to proteins
    # proteins_list = df_prots['MALDI m/z'].tolist()
    #
    # # make Gene name index
    # df_ints = df_ints.set_index('Gene name')
    # df_ints = df_ints[df_ints['MALDI m/z'].isin(proteins_list)]
    # print(df_ints)
    #
    # interval_cols = df_ints.columns.tolist()
    # interval_cols = [i for i in interval_cols if 'interval' in i]
    # df_intervals_ints = df_ints[interval_cols]
    # idx = df_intervals_ints.index.to_list()
    #
    # # min-max scale
    # # scaler = MinMaxScaler()
    # # df_intervals_ints = pd.DataFrame(scaler.fit_transform(df_intervals_ints.T).T, columns=df_intervals_ints.columns,
    # #                                 index=idx)
    # # print(df_intervals_ints)
    #
    # #fig, ax = plt.subplots(figsize=(4, 10))
    # #kws = dict(cbar_kws=dict(location='bottom'), figsize=(6, 6))
    # g = sns.clustermap(df_intervals_ints, col_cluster=False, cmap='viridis', standard_scale=0,
    #                    yticklabels=1, figsize=(4, 10), metric='euclidean', method='complete')
    # #g.ax_row_dendrogram.set_visible(False)
    # #g.ax_heatmap.set_aspect('equal')
    # #g.ax_cbar.set_visible(False)
    # g.ax_heatmap.tick_params(labelsize=5)
    # plt.savefig(os.path.join(args.output_dir, 'heatmap.svg'))
    # plt.show()
    #
    # # df_intervals_ints_cols = df_intervals_ints.columns.tolist()
    # # df_intervals_ints = df_intervals_ints.sort_values(by=df_intervals_ints_cols, ascending=False)
    # # print(df_intervals_ints)
    #
    # # interval_cols = df_intervals_ints.columns.tolist()
    # # kmeans = KMeans(n_clusters=5).fit(df_intervals_ints)
    # # labels = kmeans.labels_
    # # df_intervals_ints.insert(2, 'clusters', labels)
    # # df_intervals_ints_cols = df_intervals_ints.columns.tolist()
    # # df_intervals_ints = df_intervals_ints.sort_values(by=df_intervals_ints_cols, ascending=False)
    # # print(df_intervals_ints)
    #
    # # create heatmap
    # # fig, ax = plt.subplots(figsize=(4, 10))
    # # sns.heatmap(df_intervals_ints, cmap='viridis', annot=False, square=True, xticklabels=True, yticklabels=True, ax=ax)
    # # ax.set_aspect('equal')
    # # ax.tick_params(labelsize=4)
    # # plt.savefig(os.path.join(args.output_dir, 'heatmap.svg'))
    # # plt.show()


