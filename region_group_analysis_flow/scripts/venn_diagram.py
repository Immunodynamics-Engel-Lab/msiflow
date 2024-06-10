import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3_unweighted, venn2_unweighted, venn2, venn3


def plot_venn3(label_A, label_B, label_C, A, B, C, title, output_file='', plot=False, weighted=True):
    """
    Plots a Venn diagram of 3 sets

    :param label1: label of first set
    :param label2: label of second set
    :param label3: label of third set
    :param data1: first data set
    :param data2: second data set
    :param data3: third data set
    :param title: title of figure
    :param output_file: output file to save plot
    :param plot: set to True to plot Venn diagram
    """
    label_A = label_A + '\n({})'.format(len(A))
    label_B = label_B + '\n({})'.format(len(B))
    label_C = label_C + '\n({})'.format(len(C))
    inter_A_B = A & B
    inter_A_C = A & C
    inter_B_C = B & C
    inter_A_B_C = A & B & C
    union_A_B = A.union(B)
    union_A_C = A.union(C)
    union_B_C = B.union(C)

    # print(inter_A_B)
    # print(inter_A_C)
    # print(inter_B_C)
    # print(len(A.difference(union_B_C)))
    # print(len(B.difference(union_A_C)))
    # print(len(inter_A_B.difference(inter_A_B_C)))
    # print(len(C.difference(union_A_B)))
    # print(len(inter_A_C.difference(B)))
    # print(len(inter_B_C.difference(A)))
    # print(len(inter_A_B_C))

    if weighted:
        venn3(subsets=(len(A.difference(union_B_C)), len(B.difference(union_A_C)),
                       len(inter_A_B.difference(inter_A_B_C)), len(C.difference(union_A_B)),
                       len(inter_A_C.difference(B)), len(inter_B_C.difference(A)), len(inter_A_B_C)),
              set_labels=(label_A, label_B, label_C))
    else:
        venn3_unweighted(subsets=(len(A.difference(union_B_C)), len(B.difference(union_A_C)),
                                  len(inter_A_B.difference(inter_A_B_C)), len(C.difference(union_A_B)),
                                  len(inter_A_C.difference(B)), len(inter_B_C.difference(A)), len(inter_A_B_C)),
                         set_labels=(label_A, label_B, label_C), set_colors=('#9467bd', '#d62728', '#1f77b4'))
    plt.title(title)
    if output_file != '':
        plt.savefig(output_file)
    if plot:
        plt.show()
    plt.close()


def plot_venn2(label1, label2, data1, data2, title, output_file='', plot=False, weighted=True):
    """
    Plots a Venn diagram of 2  sets

    :param label1: label of first set
    :param label2: label of second set
    :param data1: first data set
    :param data2: second data set
    :param title: title of figure
    :param output_file: output file to save plot
    :param plot: set to True to plot Venn diagram
    """
    label1 = label1 + '\n({})'.format(len(data1))
    label2 = label2 + '\n({})'.format(len(data2))
    inter_1_2 = data1 & data2
    data1 = data1.difference(inter_1_2)
    data2 = data2.difference(inter_1_2)

    if weighted:
        venn2(subsets=(len(data1), len(data2), len(inter_1_2)), set_labels=(label1, label2))
    else:
        venn2_unweighted(subsets=(len(data1), len(data2), len(inter_1_2)), set_labels=(label1, label2))

    plt.title(title)
    if output_file != '':
        plt.savefig(output_file)
    if plot:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots common m/z of 2 to 3 files as Venn diagram')
    parser.add_argument('file1', type=str, help='first file')
    parser.add_argument('file2', type=str, help='second file')
    parser.add_argument('label1', type=str, help='label of first data')
    parser.add_argument('label2', type=str, help='label of second data')
    parser.add_argument('output_dir', type=str, help='directory to save results')
    parser.add_argument('-file3', type=str, default='', help='third file')
    parser.add_argument('-label3', type=str, default='', help='label of second data')
    parser.add_argument('-title', type=str, default='', help='title of Venn-diagram')
    parser.add_argument('-column1', type=str, default='', help='column of first file for merging')
    parser.add_argument('-column2', type=str, default='', help='column of second file for merging')
    parser.add_argument('-column3', type=str, default='', help='column of thrid file for merging')
    parser.add_argument('-delimiter1', type=str, default=',', help='first file delimiter')
    parser.add_argument('-delimiter2', type=str, default=',', help='second file delimiter')
    parser.add_argument('-delimiter3', type=str, default=',', help='third file delimiter')
    parser.add_argument('-n', type=int, default=None, help='number of rows to take from files')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df1 = pd.read_csv(args.file1, delimiter=args.delimiter1)
    df2 = pd.read_csv(args.file2, delimiter=args.delimiter2)
    #print(df1)
    #print(df2)
    if args.column1 == '':
        args.column1 = df1.columns[0]
    if args.column2 == '':
        args.column2 = df2.columns[0]
    df1[args.column1] = df1[args.column1].astype(np.float32)
    df2[args.column2] = df2[args.column2].astype(np.float32)
    # df1 = df1.round(2)
    # df2 = df2.round(2)
    if args.n:
        df1 = df1.head(args.n)
        df2 = df2.head(args.n)

    # df1 = df1.round({'Unnamed: 0': 1})
    # df2 = df2.round({'Unnamed: 0': 1})
    # print(df1)
    # print(df2)

    data1 = set(df1[args.column1].to_list())
    data2 = set(df2[args.column2].to_list())
    if args.file3 != '':
        df3 = pd.read_csv(args.file3, delimiter=args.delimiter3)
        if args.column3 == '':
            args.column3 = df3.columns[0]
        if args.n:
            df3 = df3.head(args.n)
        df3[args.column3] = df3[args.column3].astype(np.float32)
        # print(df3)
        data3 = set(df3[args.column3].to_list())
        plot_venn3(label_A=args.label1, label_B=args.label2, label_C=args.label3,
                   A=data1, B=data2, C=data3, title=args.title, plot=args.plot, weighted=True,
                   output_file=os.path.join(args.output_dir, 'venn_diagram.svg'))
        intersection = list(data1 & data2 & data3)
        data1_spec = list(data1.difference(data2.union(data3)))
        data2_spec = list(data2.difference(data1.union(data3)))
        data3_spec = list(data3.difference(data1.union(data2)))
        d1 = pd.DataFrame({args.label1 + ' spec. molecules': data1_spec})
        d2 = pd.DataFrame({args.label2 + ' spec. molecules': data2_spec})
        d3 = pd.DataFrame({args.label3 + ' spec. molecules': data3_spec})
    else:
        plot_venn2(label1=args.label1, label2=args.label2, data1=data1, data2=data2, title=args.title, plot=args.plot,
                   output_file=os.path.join(args.output_dir, 'venn_diagram.svg'))
        intersection = list(data1 & data2)
        data1_unique = list(data1.difference(data2))
        data2_unique = list(data2.difference(data1))
        data1_unique_df = pd.DataFrame({'Data 1': data1_unique})
        data2_unique_df = pd.DataFrame({'Data 2': data2_unique})
        data1_unique_df.to_csv(os.path.join(args.output_dir, args.label1 + '_specific_molecules.csv'))
        data2_unique_df.to_csv(os.path.join(args.output_dir, args.label2 + '_specific_molecules.csv'))


    common_df = pd.DataFrame({'Common molecules': intersection})
    # print(common_df)
    common_df.to_csv(os.path.join(args.output_dir, 'common_molecules.csv'))
    if args.file3 != '':
        d1.to_csv(os.path.join(args.output_dir, args.label1 + '_specific_molecules.csv'))
        d2.to_csv(os.path.join(args.output_dir, args.label2 + '_specific_molecules.csv'))
        d3.to_csv(os.path.join(args.output_dir, args.label3 + '_specific_molecules.csv'))

