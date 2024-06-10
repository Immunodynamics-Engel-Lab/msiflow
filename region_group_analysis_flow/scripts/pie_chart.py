import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots pie chart')
    parser.add_argument('input', type=str, help='csv file with lipid classes')
    parser.add_argument('output', type=str, help='output file with pie chart')
    parser.add_argument('-col', type=str, default='Lipid class', help='column name')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    # read data
    df = pd.read_csv(args.input, index_col=0, delimiter=',')

    if not df.empty:
        # value counts of defined column
        val_counts = df[args.col].value_counts()
        data = val_counts.to_numpy()
        labels = val_counts.index.to_numpy()

        # plot pie chart
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        wedges, texts, autotexts = ax.pie(data, labels=labels, autopct=lambda pct: func(pct, data),
                                          textprops=dict(color="k"))
        ax.legend(wedges, labels)

    plt.savefig(args.output)

    if args.plot:
        plt.show()