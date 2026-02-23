import pandas as pd
import numpy as np
import argparse
import os
from scipy.stats import anderson, levene, ranksums, shapiro, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import re


def decode_names(names):
    regions = set()
    samples = set()
    for f in names:
        regions.add(f.split('_')[0])
        samples.add(f.split('_')[1] + '_' + f.split('_')[2])
    return list(regions), list(samples)


def plot_sample_percentage(percentage_per_row, fc_thr, sample_perc_thr, region_1, region_2, high=True, plot=False):
    fc_sample_percentage_df = percentage_per_row.reset_index()
    fc_sample_percentage_df.columns = ['m/z', 'sample percentage']
    fc_sample_percentage_df = fc_sample_percentage_df.sort_values(by=['sample percentage'], ascending=False)
    fc_sample_percentage_df['m/z'] = fc_sample_percentage_df['m/z'].round(2)
    if len(fc_sample_percentage_df) > 20:
        fc_sample_percentage_df = fc_sample_percentage_df.head(20)  # take the first 20 rows
    print(fc_sample_percentage_df)

    fig, ax = plt.subplots(figsize=(7, 10))
    sns.barplot(data=fc_sample_percentage_df, x='sample percentage', y='m/z', orient='h',
                order=fc_sample_percentage_df['m/z'], ax=ax)
    # sns.barplot(data=fc_sample_percentage_df, x='sample percentage', y='m/z', orient='h',
    #             order=fc_sample_percentage_df['sample percentage'])
    if high:
        plt.title('sample percentage of m/z with log2(FC) > {0} ({1} vs {2})'.format(fc_thr, region_1, region_2))
    else:
        plt.title('sample percentage of m/z with log2(FC) < -{0} ({1} vs {2})'.format(fc_thr, region_1, region_2))
    ax.set_xlabel('samples (%)')
    ax.set_ylabel('m/z')
    if high:
        fl_name = '{}_fc_{}_sample_percentage'.format(fc_thr, sample_perc_thr)
    else:
        fl_name = '-{}_fc_{}_sample_percentage'.format(fc_thr, sample_perc_thr)
    fc_sample_percentage_df.to_csv(os.path.join(args.output_dir, fl_name + '.csv'))
    plt.savefig(os.path.join(args.output_dir, fl_name + '.svg'))

    if plot:
        fig.show()
    plt.close(fig)


def mean_fc_bar_plot(df, n, output_file, output_csv, annot_fl='', plot=False):
    if annot_fl != '':
        annot_df = pd.read_csv(annot_fl, delimiter=',', index_col=0)

        annot_df.index = annot_df.index.astype(float).round(4)
        df.index = df.index.astype(float).round(4)

        common = df.index.intersection(annot_df.index)
        df_join = df.loc[common].join(annot_df.loc[common])

        y_col = 'Name'
        top_rows = df_join.nlargest(n, 'row_mean_above_thresh').reset_index()

    else:
        y_col = 'index'
        top_rows = df.nlargest(n, 'row_mean_above_thresh').reset_index()

    if annot_fl == '':
        top_rows['index'] = top_rows['index'].astype(str)
    else:
        top_rows.drop(columns=['index'], inplace=True)

    top_rows = top_rows.sort_values('row_mean_above_thresh', ascending=False)
    top_rows.to_csv(output_csv)
    print("top_rows=", top_rows)

    # seaborn bar plot
    fig, ax = plt.subplots(figsize=(7, 10))
    sns.barplot(y=y_col, x='row_mean_above_thresh', data=top_rows, ax=ax)
    ax.set_ylabel('m/z')
    ax.set_xlabel('mean log2(FC)')
    plt.subplots_adjust(left=0.25)  # Increase left margin
    plt.savefig(output_file)

    if plot:
        fig.show()
    plt.close(fig)


def plot_best_fc_sample(df, x_col, smpl, regions, out_dir):
    fig, ax = plt.subplots(figsize=(7, 10))
    sns.barplot(x=x_col, y=df.index, data=df, orient='h', order=df.index.to_list(), ax=ax)
    ax.set_xlabel('log2(FC)')
    ax.set_ylabel('m/z')
    plt.title('{0} {1} vs {2}'.format(smpl, regions[0], regions[1]))

    plt.savefig(os.path.join(out_dir, 'FC_{0}_{1}_vs_{2}.svg'.format(smpl, regions[0], regions[1])),
                transparent=True)

    if args.plot:
        fig.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculates fold changes between two regions per individual MSI sample')
    parser.add_argument('input_file', type=str, help='input file with summarized intensities per region')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot volcano')
    parser.add_argument('-fc_thr', type=float, default=0.5,
                        help='absolute log2 FC threshold to summarize over all samples')
    parser.add_argument('-sample_perc_thr', type=float, default=0.5,
                        help='sample percentage in which the FC threshold needs to be met')
    parser.add_argument('-annotations_file', type=str, default='',
                        help='optional file containing annotated m/z values')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df = pd.read_csv(args.input_file, index_col=0)
    cols = df.columns

    print(df)
    print(cols)

    # remove m/z rows where any value is 0
    df = df.loc[~(df.eq(0).any(axis=1))]

    regions, samples = decode_names(cols)
    print(regions, samples)

    for smpl in samples:
        # calculate FC for sample
        col1 = regions[0] + '_' + smpl
        col2 = regions[1] + '_' + smpl
        res_col = 'FC_' + regions[0] + '_' + regions[1] + '_' + smpl
        df[res_col] = np.log2(df[col1] / df[col2])

        # get m/z with highest and lowest FC
        smpl_cols = [col for col in df.columns if smpl in col]
        smpl_df = df[smpl_cols]
        print(smpl_df)

        top_10_largest = smpl_df.nlargest(10, res_col)
        top_10_smallest = smpl_df.nsmallest(10, res_col)
        filtered_smpl_df = pd.concat([top_10_largest, top_10_smallest])

        filtered_smpl_df.sort_values(by=[res_col], ascending=False, inplace=True)
        filtered_smpl_df.index = filtered_smpl_df.index.round(2)
        print(filtered_smpl_df)

        filtered_smpl_df.to_csv(os.path.join(args.output_dir, res_col + '.csv'))

        # plot best FC of sample
        plot_best_fc_sample(df=filtered_smpl_df, x_col=res_col, smpl=smpl, regions=regions, out_dir=args.output_dir)

    print(df)
    df.to_csv(os.path.join(args.output_dir, 'FC.csv'))

    # filter df based on fc in specific proportion of samples

    fc_cols = [col for col in df.columns if 'FC' in col]
    fc_df = df[fc_cols]
    print(fc_df)

    threshold = args.sample_perc_thr * fc_df.shape[1]  # sample percentage of the number of columns
    filtered_df = fc_df[
        (fc_df.gt(args.fc_thr).sum(axis=1) >= threshold) | (fc_df.lt(-args.fc_thr).sum(axis=1) >= threshold)]

    filtered_high_df = fc_df[(fc_df.gt(args.fc_thr).sum(axis=1) >= threshold)]
    filtered_low_df = fc_df[(fc_df.lt(-args.fc_thr).sum(axis=1) >= threshold)]

    print("threshold=", threshold)
    print("filtered_df=", filtered_df)

    # Count the number of columns per row with value > 1 or < -1
    count_per_row_high = (filtered_high_df > args.fc_thr).sum(axis=1)
    count_per_row_low = (filtered_low_df < -args.fc_thr).sum(axis=1)

    print(count_per_row_high)
    print(count_per_row_low)

    percentage_per_row_high = (filtered_high_df > args.fc_thr).sum(axis=1) / filtered_df.shape[1] * 100
    percentage_per_row_low = (filtered_low_df < -args.fc_thr).sum(axis=1) / filtered_df.shape[1] * 100

    # plot
    plot_sample_percentage(percentage_per_row_high, args.fc_thr, args.sample_perc_thr, regions[0], regions[1], True,
                           args.plot)
    plot_sample_percentage(percentage_per_row_low, args.fc_thr, args.sample_perc_thr, regions[0], regions[1], False,
                           args.plot)

    # save filtered df
    filtered_high_df.to_csv(
        os.path.join(args.output_dir, '{}_fc_{}_samples_filtered.csv'.format(args.fc_thr, args.sample_perc_thr)))
    filtered_low_df.to_csv(
        os.path.join(args.output_dir, '-{}_fc_{}_samples_filtered.csv'.format(args.fc_thr, args.sample_perc_thr)))

    # save filtered df with annotation
    if args.annotations_file != '':
        annot_df = pd.read_csv(args.annotations_file, delimiter=',', index_col=0)

        annot_df.index = annot_df.index.astype(float).round(4)
        filtered_high_df.index = filtered_high_df.index.astype(float).round(4)
        filtered_low_df.index = filtered_low_df.index.astype(float).round(4)

        common_high = filtered_high_df.index.intersection(annot_df.index)
        common_low = filtered_low_df.index.intersection(annot_df.index)

        filtered_high_df_join = filtered_high_df.loc[common_high].join(annot_df.loc[common_high])
        filtered_low_df_join = filtered_low_df.loc[common_low].join(annot_df.loc[common_low])

        filtered_high_df_join.to_csv(os.path.join(args.output_dir,
                                                  '{}_fc_{}_samples_filtered_annot.csv'.format(args.fc_thr,
                                                                                               args.sample_perc_thr)))
        filtered_low_df_join.to_csv(os.path.join(args.output_dir,
                                                 '-{}_fc_{}_samples_filtered_annot.csv'.format(args.fc_thr,
                                                                                               args.sample_perc_thr)))

    # calculate mean fc of samples with fc above threshold
    filtered_high_df['row_mean_above_thresh'] = filtered_high_df.where(filtered_high_df > args.fc_thr).mean(axis=1)
    filtered_low_df['row_mean_above_thresh'] = filtered_low_df.where(filtered_low_df < -args.fc_thr).mean(axis=1)

    print("filtered_high_df=", filtered_high_df)
    print("filtered_low_df=", filtered_low_df)

    # generate mean fc bar plot of top N
    top_n = 20
    mean_fc_bar_plot(df=filtered_high_df,
                     n=top_n,
                     output_file=os.path.join(args.output_dir, '{}_fc_{}_samples_top_mean_FC.svg'.format(args.fc_thr,
                                                                                                         args.sample_perc_thr)),
                     output_csv=os.path.join(args.output_dir, '{}_fc_{}_samples_top_mean_FC.csv'.format(args.fc_thr,
                                                                                                        args.sample_perc_thr)),
                     annot_fl=args.annotations_file,
                     plot=args.plot)
    mean_fc_bar_plot(df=filtered_low_df,
                     n=top_n,
                     output_file=os.path.join(args.output_dir, '-{}_fc_{}_samples_top_mean_FC.svg'.format(args.fc_thr,
                                                                                                          args.sample_perc_thr)),
                     output_csv=os.path.join(args.output_dir, '-{}_fc_{}_samples_top_mean_FC.csv'.format(args.fc_thr,
                                                                                                         args.sample_perc_thr)),
                     annot_fl=args.annotations_file,
                     plot=args.plot)


