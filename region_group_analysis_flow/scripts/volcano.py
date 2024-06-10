import pandas as pd
import numpy as np
import argparse
import os
from scipy.stats import anderson, levene, ranksums, shapiro, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt


def calc_log2_fold_change(gr_1, gr_2, log2=True):
    group_1_means = gr_1.mean(axis=1)
    group_2_means = gr_2.mean(axis=1)
    fold_change = np.divide(group_1_means, group_2_means)
    if log2:
        return np.log2(fold_change)
    else:
        return fold_change


def calc_p_value(gr_1, gr_2):
    result = []
    for x in range(0, gr_1.shape[0]):
        grp1, grp2 = gr_1[x], gr_2[x]
        var = VarianceTest(grp1, grp2)
        if grp1.shape[0] > 2 and grp2.shape[0] > 2:
            normal = NormalityTest(grp1, grp2)
        else:
            normal = False
        if normal:
            _, pval = ttest_ind(grp1, grp2, equal_var=var, nan_policy='omit')
        else:
            _, pval = ranksums(grp1, grp2)
        result.append(pval)
    return np.array(result)


def calc_snr(gr_1, gr_2):
    mean = np.mean(gr_1, axis=1) - np.mean(gr_2, axis=1)
    std = np.std(gr_1, axis=1) - np.std(gr_2, axis=1)
    snr = mean / std
    return snr


def NormalityTest(grp1: np.ndarray, grp2: np.ndarray):
    """
    Tests if feature was drawn from normal distribution (Shapiro).
    :param grp1: ndarray, shape(n_samples,).
    :param grp2: ndarray, shape(n_samples,).
    :return: True: Null hypothesis accepted, data was drawn from normal distribution.
             False: Null hypothesis rejected, data was not drawn from normal distribution.
    """
    _, sp = np.float32(shapiro(grp1))
    _, sp2 = np.float32(shapiro(grp2))
    sp_mean = np.mean([sp, sp2], dtype=np.float16)

    if sp_mean > 0.05:
        return True
    else:
        return False


def VarianceTest(grp1: np.ndarray, grp2: np.ndarray):
    """
    Tests if feature indicates homogeneity of variance (Levene).
    :param grp1: ndarray, shape(n_samples,).
    :param grp2: ndarray, shape(n_samples,).
    :return: True: Null hypothesis accepted, input samples are from populations with equal variance.
             False: Null hypothesis rejected, input samples are from populations with unequal variance.
    """
    _, vp = np.float16(levene(grp1, grp2, center='mean'))
    # Null Hypothesis: all input samples are from populations with equal variance
    if vp > 0.05:
        return True
    else:
        return False


def volcano_plot(input, output_dir, gr1, gr2, plot=True, transpose=False, fc_thr=1, ttest=False,
                 protein_level=True, proteins_of_interest='', color_col='', categorical=False, dot_size=10):
    annot_prefix = 'non_annot'
    if color_col != '':
        annot_prefix = 'annot'
    df_data = pd.read_csv(input)
    df_data = df_data.fillna(0)
    # print(df_data)
    if transpose:
        spectra = df_data.iloc[:, 1:].to_numpy()
        mzs = df_data.columns[1:].to_numpy().astype(float)
        idx = df_data['ID'].to_list()

        df_data = pd.DataFrame(columns=mzs, data=spectra, index=idx)
        df_data = df_data.transpose()

    gr1_cols = [x for x in df_data.columns if gr1 in x]
    gr2_cols = [x for x in df_data.columns if gr2 in x]

    # filter out rows where one group only contains zero values
    df_data['gr1_sum'] = df_data[gr1_cols].sum(axis=1)
    df_data['gr2_sum'] = df_data[gr2_cols].sum(axis=1)
    #df_data = df_data.loc[(df_data['gr1_sum'] > 0) & (df_data['gr2_sum'] > 0)]

    arr_gr1 = df_data[gr1_cols].to_numpy().astype(float)
    arr_gr2 = df_data[gr2_cols].to_numpy().astype(float)

    # print(df_data)

    # calculate p-value and FC
    if ttest:
        _, p = ttest_ind(arr_gr1, arr_gr2, axis=1, equal_var=True, nan_policy='omit')  # standard t-test
    else:
        p = calc_p_value(arr_gr1, arr_gr2)
    fold_change = calc_log2_fold_change(arr_gr1, arr_gr2, log2=True)
    snr = calc_snr(arr_gr1, arr_gr2)
    df_data['p-value'] = p
    df_data['p-value'] = -np.log10(p)
    df_data['Fold change'] = fold_change
    df_data['SNR'] = snr
    #df_data = df_data.dropna(axis=0)

    # keep only peptide with highest p-value for one protein
    if protein_level:
        idx = df_data.groupby(['Accession'])['p-value'].transform(max) == df_data['p-value']
        df_data = df_data[idx]

    # get significance
    if fc_thr != 0:
        conditions = [(df_data['p-value'] >= 1.3) & (df_data['Fold change'] > fc_thr),
                      (df_data['p-value'] >= 1.3) & (df_data['Fold change'] < -fc_thr)]
    else:
        conditions = [(df_data['p-value'] >= 1.3) & (df_data['Fold change'] > fc_thr),
                      (df_data['p-value'] >= 1.3) & (df_data['Fold change'] < fc_thr)]
    choices = [1, -1]
    df_data['significance'] = np.select(conditions, choices, default=0)
    # plot volcano
    plt.figure(figsize=(8, 6))
    if proteins_of_interest != '':
        #df_poi = pd.read_csv(proteins_of_interest, delimiter='\t')
        df_poi = pd.read_csv(proteins_of_interest)
        poi = df_poi.iloc[:, 0].to_numpy()
        # print(poi)
        #accession = df_data['Accession'].to_numpy()
        accession = df_data.iloc[:, 0].to_numpy()
        is_poi = 1*np.isin(accession, poi)
        df_data['Proteins of interest'] = is_poi
        # print(df_data[df_data['Proteins of interest'] == True])

        fig = sns.scatterplot(data=df_data[df_data['significance'] == 0], x="Fold change", y="p-value", s=dot_size, alpha=0.5,
                              palette=['lightgray'], edgecolor='black', hue='significance', legend=False)
        sns.scatterplot(data=df_data[df_data['significance'] == -1], x="Fold change", y="p-value", s=dot_size, alpha=0.5,
                        palette=['lightsteelblue'], edgecolor='black', hue='significance', legend=False)
        sns.scatterplot(data=df_data[(df_data['Proteins of interest'] == 1) & (df_data['significance'] == -1)],
                        x="Fold change", y="p-value", s=dot_size, alpha=1, hue='Proteins of interest',
                        palette=['steelblue'], edgecolor='black', legend=False)
        sns.scatterplot(data=df_data[df_data['significance'] == 1], x="Fold change", y="p-value", s=dot_size, alpha=0.5,
                        palette=['mistyrose'], edgecolor='black', hue='significance', legend=False)
        sns.scatterplot(data=df_data[(df_data['Proteins of interest'] == 1) & (df_data['significance'] == 1)],
                        x="Fold change", y="p-value", s=dot_size, alpha=1,
                        palette=['red'], edgecolor='black', hue='Proteins of interest', legend=False)
    elif color_col != '':
        if categorical:
            #custom_palette = ["#3498db", "#2ecc71", "#e74c3c"]
            #sns.set_palette(custom_palette)
            # fig = sns.scatterplot(data=df_data[df_data['significance'] == 0], x="Fold change", y="p-value", s=dot_size,
            #                     alpha=0.5, edgecolor='black', palette='tab10', hue=color_col, legend=False)
            # sns.scatterplot(data=df_data[df_data['significance'] != 0], x="Fold change", y="p-value", s=dot_size,
            #                      alpha=1.0, edgecolor='black', palette='tab10', hue=color_col)


            # fig = sns.scatterplot(data=df_data[df_data[color_col] == 0 & (df_data['significance'] == 0)], x="Fold change", y="p-value", s=dot_size,
            #                 alpha=0.5, edgecolor='black', linewidth=0, color='grey', legend=False)
            fig = sns.scatterplot(data=df_data[(df_data[color_col] == 0) & (df_data['significance'] != 0)], x="Fold change",
                            y="p-value", s=dot_size, edgecolor='black', linewidth=0, color='grey', legend=True)
            sns.scatterplot(data=df_data[(df_data[color_col] != 0) & (df_data['significance'] != 0)], x="Fold change",
                                  y="p-value", s=dot_size, alpha=1.0, edgecolor='black', linewidth=0, palette='tab10', hue=color_col,
                                  legend=True)
            sns.scatterplot(data=df_data[(df_data[color_col] == 0) & (df_data['significance'] == 0)], x="Fold change",
                            y="p-value", s=dot_size,
                            alpha=0.5, color='grey', linewidth=0, edgecolor='black', legend=False)
            # sns.scatterplot(data=df_data[(df_data[color_col] != 0) & (df_data['significance'] == 0)], x="Fold change", y="p-value", s=dot_size,
            #                alpha=0.5, palette='tab10', hue=color_col, linewidth=0, edgecolor='black', legend=False)
            sns.scatterplot(data=df_data[(df_data[color_col] != 0) & (df_data['significance'] == 0)], x="Fold change", y="p-value", s=dot_size,
                           alpha=0.5, palette='tab10', color='grey', linewidth=0, edgecolor='black', legend=False)

        else:
            #ax = sns.scatterplot(x="Fold change", y="p-value", hue=color_col, palette='Reds', data=df_data)
            sns.scatterplot(data=df_data[df_data['significance'] == 0], x="Fold change", y="p-value", s=dot_size,
                                 alpha=0.5, palette=['lightgray'], edgecolor='black', hue=color_col, legend=False)
            sns.scatterplot(data=df_data[df_data['significance'] == -1], x="Fold change", y="p-value", s=dot_size,
                            alpha=1.0, palette='Blues', edgecolor='black', hue=color_col)
            sns.scatterplot(data=df_data[df_data['significance'] == 1], x="Fold change", y="p-value", s=dot_size, alpha=1.0,
                            palette='Reds', edgecolor='black', hue=color_col)

            norm = plt.Normalize(df_data[color_col].min(), df_data[color_col].max())
            sm1 = plt.cm.ScalarMappable(cmap="Reds_r", norm=norm)
            sm1.set_array([])
            sm2 = plt.cm.ScalarMappable(cmap="Blues_r", norm=norm)
            sm2.set_array([])

            # Remove the legend and add a colorbar
            fig.get_legend().remove()
            #fig.figure.colorbar(sm1)
            #fig.figure.colorbar(sm2)
    else:
        fig = sns.scatterplot(data=df_data[df_data['significance'] == 0], x="Fold change", y="p-value", s=dot_size, alpha=0.5,
                              palette=['lightsteelblue'], edgecolor='black', hue='significance', legend=False)
        sns.scatterplot(data=df_data[df_data['significance'] == 1], x="Fold change", y="p-value", s=dot_size, alpha=1,
                        palette=['red'], edgecolor='black', hue='significance', legend=False)
        sns.scatterplot(data=df_data[df_data['significance'] == -1], x="Fold change", y="p-value", s=dot_size, alpha=1,
                        palette=['steelblue'], edgecolor='black', hue='significance', legend=False)

    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(xmin, ymax+0.1, gr1 + ' / ' + gr2)
    plt.ylabel('$-log_{10}$(p-value)')
    plt.xlabel('$log_{2}$(fold change)')
    fig.axhline(1.3, color='k', linestyle='--', lw=0.8)
    if fc_thr != 0:
        fig.axvline(-fc_thr, color='k', linestyle='--', lw=0.8)
        fig.axvline(fc_thr, color='k', linestyle='--', lw=0.8)
    else:
        fig.axvline(0, color='k', lw=0.3)

    plt.savefig(os.path.join(output_dir, annot_prefix + '_volcano_plot.svg'))
    if plot:
        plt.show()
    plt.close()

    # save significant data file with p-value and fold change
    df_sig_up = df_data.loc[(df_data['p-value'] > 1.3) & (df_data['Fold change'] > fc_thr)]
    if fc_thr != 0:
        df_sig_down = df_data.loc[(df_data['p-value'] > 1.3) & (df_data['Fold change'] < -fc_thr)]
    else:
        df_sig_down = df_data.loc[(df_data['p-value'] > 1.3) & (df_data['Fold change'] < 0)]
    df_sig = df_data.loc[(df_data['significance'] == 1) | (df_data['significance'] == -1)]
    #df_sig = df_data

    df_sig = df_sig.drop(columns=['significance', 'gr1_sum', 'gr2_sum'])
    df_sig_up = df_sig_up.drop(columns=['significance', 'gr1_sum', 'gr2_sum'])
    df_sig_down = df_sig_down.drop(columns=['significance', 'gr1_sum', 'gr2_sum'])
    df_data = df_data.drop(columns=['significance', 'gr1_sum', 'gr2_sum'])
    if len(gr1_cols) == 1 and len(gr2_cols) == 1:
        df_data = df_data.drop(columns=['p-value', 'SNR'])
    #print(df_sig)
    #print(df_data)

    df_data.to_csv(os.path.join(output_dir, '{}_{}_{}_analysis.csv').format(annot_prefix, gr1, gr2), index=False)
    #if not df_sig.empty:
    df_sig.to_csv(os.path.join(output_dir, '{}_{}_{}_regulated.csv').format(annot_prefix, gr1, gr2), index=False)
    df_sig_up.to_csv(os.path.join(output_dir, '{}_{}_{}_upregulated.csv').format(annot_prefix, gr1, gr2), index=False)
    df_sig_down.to_csv(os.path.join(output_dir, '{}_{}_{}_downregulated.csv').format(annot_prefix, gr1, gr2), index=False)



    # org_len = df_data.shape[0]
    # # keep only best protein for each peptide
    # #df_data = df_data.groupby("Accession", as_index=False).max()
    #
    # # for the same peptide keep the one with highest NSAF
    # #idx = combined.groupby(['Peptide_mass'])['NSAF'].transform(max) == combined['NSAF']
    #
    # df_data = df_data.astype({"p-value": float})
    #
    # idx = df_data.groupby(['Accession'])['p-value'].transform(max) == df_data['p-value']
    # df_data = df_data[idx]
    # print('reduced data from {} to {} by keeping the peptide with highest p-value for each protein'.
    #       format(org_len, df_data.shape[0]))
    # print(df_data.columns)
    #
    # if fc_thr != 0:
    #     conditions = [(df_data['p-value'] >= 1.3) & (df_data['Fold change'] > fc_thr),
    #                   (df_data['p-value'] >= 1.3) & (df_data['Fold change'] < -fc_thr)]
    # else:
    #     conditions = [(df_data['p-value'] >= 1.3) & (df_data['Fold change'] > fc_thr),
    #                   (df_data['p-value'] >= 1.3) & (df_data['Fold change'] < fc_thr)]
    # choices = [1, -1]
    # df_data['significance'] = np.select(conditions, choices, default=0)
    #
    # # plot volcano
    # plt.figure(figsize=(8, 6))
    # fig = sns.scatterplot(data=df_data[df_data['significance'] == 0], x="Fold change", y="p-value", s=7, alpha=0.2,
    #                        palette=['lightsteelblue'], edgecolor='black', hue='significance', legend=False)
    # sns.scatterplot(data=df_data[df_data['significance'] == 1], x="Fold change", y="p-value", s=7, alpha=1,
    #                        palette=['red'], edgecolor='black', hue='significance', legend=False)
    # sns.scatterplot(data=df_data[df_data['significance'] == -1], x="Fold change", y="p-value", s=7, alpha=1,
    #                 palette=['steelblue'], edgecolor='black', hue='significance', legend=False)
    #
    # fig.axhline(1.3, color='k', linestyle='--', lw=0.8)
    # if fc_thr != 0:
    #     fig.axvline(-1, color='k', linestyle='--', lw=0.8)
    #     fig.axvline(1, color='k', linestyle='--', lw=0.8)
    # else:
    #     fig.axvline(0, color='k', lw=0.3)
    #
    # plt.savefig(os.path.join(output_dir, 'volcano_plot.pdf'))
    # if plot:
    #     plt.show()
    # plt.close()

    # # save significant data file with p-value and fold change
    # if regulation == 'up':
    #     df_sig2 = df_data.loc[(df_data['p-value'] > 1.3) & (df_data['Fold change'] > fc_thr)]
    # elif regulation == 'down':
    #     if fc_thr != 0:
    #         df_sig2 = df_data.loc[(df_data['p-value'] > 1.3) & (df_data['Fold change'] < -fc_thr)]
    #     else:
    #         df_sig2 = df_data.loc[(df_data['p-value'] > 1.3) & (df_data['Fold change'] < 0)]
    # else:
    #     df_sig2 = df_data.loc[(df_data['significance'] == 1) | (df_data['significance'] == -1)]
    #
    # df_sig2 = df_sig2.drop(columns=['significance', 'gr1_sum', 'gr2_sum'])
    # df_sig2.to_csv(os.path.join(output_dir, '{}_{}_{}regulated_proteins.csv').format(gr1, gr2, regulation), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates volcano plot')
    parser.add_argument('input_file', type=str, help='input file')
    parser.add_argument('group_1', type=str, help='name of group 1')
    parser.add_argument('group_2', type=str, help='name of group 2')
    parser.add_argument('-output_dir', type=str, default='', help='directory to save volcano plot and data table')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot volcano')
    parser.add_argument('-transpose', type=bool, default=False, help='set to True if m/z values are row indices and '
                                                                     'group names are columns')
    parser.add_argument('-fc_thr', type=float, default=0, help='threshold for fold change')
    parser.add_argument('-ttest', type=bool, default=False, help='set to True for standard t-test')
    parser.add_argument('-protein_level', type=bool, default=False, help='set to True to plot volcano on protein level')
    parser.add_argument('-molecules_of_interest', type=str, default='', help='csv file path with list of molecules of '
                                                                            'interest which should be highlighted in '
                                                                            'the volcano')
    parser.add_argument('-color_col', type=str, default='', help='column name containing colors for dots')
    parser.add_argument('-categorical', type=bool, default=False, help='set to True if color col is categorical')
    parser.add_argument('-dot_size', type=int, default=50, help='size of dots in volcano plot')
    args = parser.parse_args()

    if args.output_dir == '':
        args.output_dir = os.path.join(os.path.dirname(args.input_file), 'FC' + str(args.fc_thr))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    volcano_plot(args.input_file, args.output_dir, args.group_1, args.group_2, args.plot, args.transpose,
                 args.fc_thr, args.ttest, args.protein_level, args.molecules_of_interest, args.color_col,
                 args.categorical, args.dot_size)
