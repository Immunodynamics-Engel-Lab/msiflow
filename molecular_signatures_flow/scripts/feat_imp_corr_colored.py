import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os


def plot_ranking(df, y_col, outfile, norm, plot=False):
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=df['feature importance'], y=df[y_col], orient='h', order=df[y_col], hue=df['mean'],
                     palette=newcmp, dodge=False, hue_norm=norm)
    ax.get_legend().remove()
    ax.figure.colorbar(sm, ax=ax)
    plt.tight_layout()
    plt.savefig(outfile)

    if plot:
        plt.show()

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot top n features color coded by its correlation')
    parser.add_argument('feat_imp_file', type=str, help='first file path containing feature importance')
    parser.add_argument('corr_file', type=str, help='second file path containing correlation')
    parser.add_argument('-output_dir', type=str, default='', help='directory to save results')
    parser.add_argument('-annot_file', type=str, default='', help='file to annotate m/z values')
    parser.add_argument('-cmap', type=str, default='Reds', help='cmap')
    parser.add_argument('-n', type=int, default=10, help='number of features to plot')
    parser.add_argument('-corr_thr', type=float, default=None,
                        help='set a value to only plot m/z with correlation above this threshold')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot')
    args = parser.parse_args()

    if args.output_dir == '':
        args.output_dir = os.path.join(os.path.dirname(args.feat_imp_file), 'combined')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df_feat_imp = pd.read_csv(args.feat_imp_file)
    df_corr = pd.read_csv(args.corr_file)

    if args.corr_thr:
        df_corr = df_corr.loc[df_corr['mean'] > args.corr_thr]

    # print(df_feat_imp)
    # print(df_corr)

    df_feat_imp[df_feat_imp.columns[0]] = df_feat_imp[df_feat_imp.columns[0]].astype(np.float32)
    df_corr[df_corr.columns[0]] = df_corr[df_corr.columns[0]].astype(np.float32)
    #
    # cols_df2 = [df2.columns[0]]
    # cols_df2.extend(args.cols_sec_file)
    #
    # df2 = df2[cols_df2]
    #
    # print(df1)
    # print(df2)
    #
    merged = pd.merge(df_feat_imp, df_corr, left_on=df_feat_imp.columns[0], right_on=df_corr.columns[0], how='inner')
    merged = merged[['m/z', 'feature importance', 'mean']]
    # print(merged)

    #top_feat_imp_df = merged.iloc[:args.n, :]
    top_feat_imp_df = merged
    top_mzs = top_feat_imp_df['m/z'].to_numpy()
    # print(top_feat_imp_df)

    if args.annot_file != '':
        df_annot = pd.read_csv(args.annot_file, delimiter='\t')
        df_annot[df_annot.columns[0]] = df_annot[df_annot.columns[0]].astype(np.float32)
        df_annot = df_annot[['Input Mass', 'Name']]
        df_annot.set_index('Input Mass', inplace=True, drop=True)
        df_annot = df_annot.groupby(df_annot.index)['Name'].apply(lambda x: ','.join(x)).to_frame()
        top_feat_imp_df = pd.merge(top_feat_imp_df, df_annot, left_on='m/z', right_on='Input Mass')


    top_feat_imp_df['m/z'] = np.round(top_feat_imp_df['m/z'].to_numpy().astype(np.float64), 2)
    top_feat_imp_df['m/z'] = top_feat_imp_df['m/z'].to_numpy().astype(str)

    top_feat_imp_df = top_feat_imp_df.iloc[:args.n, :]
    # print(top_feat_imp_df)

    #top_feat_imp_df = top_feat_imp_df.sort_values(by=['mean'], ascending=False)

    # cmap
    #args.cmap = mpc.LinearSegmentedColormap.from_list("", ["#FFFFFF", "#5D4FA2"])
    top = cm.get_cmap('Greys_r', 128)
    bottom = cm.get_cmap(args.cmap, 128)

    g = top_feat_imp_df.groupby('mean')
    n = g.size()
    min_val = np.min(top_feat_imp_df['mean'].to_numpy())
    max_val = np.max(top_feat_imp_df['mean'].to_numpy())
    # print(top_feat_imp_df['mean'])
    # print(np.min(top_feat_imp_df['mean'].to_numpy()))
    # print(np.max(top_feat_imp_df['mean'].to_numpy()))
    if min_val > 0:
        norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
        newcolors = bottom(np.linspace(0, 1, 128))
    else:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
        newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                               bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    sm = plt.cm.ScalarMappable(cmap=newcmp, norm=norm)

    # print(top_feat_imp_df)

    # plot annotated ranking
    if args.annot_file != '':
        plot_ranking(top_feat_imp_df, 'Name', os.path.join(args.output_dir, 'annot_top_features.svg'),
                     norm, args.plot)

    # plot m/z ranking
    plot_ranking(top_feat_imp_df, 'm/z', os.path.join(args.output_dir, 'top_features.svg'),
                 norm, args.plot)

    # save top n features
    top_feat_imp_df.to_csv(os.path.join(args.output_dir, 'top_features.csv'), index=False)


    # #norm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    # norm = matplotlib.colors.TwoSlopeNorm(vmin=top_feat_imp_df['mean'].min(), vcenter=0, vmax=top_feat_imp_df['mean'].max())
    #
    # sns.barplot(x=top_feat_imp_df['feature importance'], y=top_feat_imp_df['m/z'], orient='h', order=top_feat_imp_df['m/z'],
    #             hue=top_feat_imp_df['mean'], palette=args.cmap, dodge=False, hue_norm=norm)
    # plt.show()
