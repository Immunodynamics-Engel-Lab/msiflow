import pandas as pd
import warnings
import os
import sys
import argparse
import numpy as np
import multiprocessing
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg.plot import plot_ion_image

warnings.filterwarnings('ignore', module='pyimzml')


def get_mean_ints_img(pyx, msi_df):
    coords = msi_df.index.tolist()
    msi_img = np.zeros(pyx).astype(np.uint8)
    for x_val, y_val in coords:
        msi_img[y_val - 1, x_val - 1] = msi_df.loc[(x_val, y_val)]
    return msi_img


if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Generate ion images of specific mz or multiple mz in csv file')
        parser.add_argument('input', type=str, help='directory or single imzML file(s)/.d directory(ies)')
        parser.add_argument('output_dir', type=str, default='', help='directory to store results')
        parser.add_argument('-mz', type=float, default=None, help='m/z value')
        parser.add_argument('-mz_file', type=str, default='', help='csv file with m/z values')
        parser.add_argument('-num', type=int, default=None, help='takes this number of top m/z values from file to save')
        parser.add_argument('-tol', default=0.01, type=float, help='tolerance')
        parser.add_argument('-unit', default='da', type=str, help='unit for tolerance, either da or ppm')
        parser.add_argument('-CLAHE', default=False, type=bool, help='set to True for CLAHE')
        parser.add_argument('-contrast_stretch', default=False, type=bool, help='set to True for contrast stretch')
        parser.add_argument('-lower', default=0.0, type=float, help='lower percentile for contrast stretching')
        parser.add_argument('-upper', default=99.0, type=float, help='upper percentile for contrast stretching')
        parser.add_argument('-plot', default=False, type=bool, help='set to True for plotting ion image')
        parser.add_argument('-cmap', default='viridis', type=str, help='colormap')
        parser.add_argument('-pyimzml', default=False, type=bool, help='if True, use pyimzml library')
        parser.add_argument('-remove_iso_px', default=False, type=bool, help='if True, isolated pixels are removed '
                                                                             'from image')
        parser.add_argument('-format', default='png', type=str, help='output file format, either png or tif')
        args = parser.parse_args()

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        # read in data
        if os.path.isdir(args.input):
            input_dir = args.input
            data = [d for d in os.listdir(args.input) if d.endswith('.imzML') or d.endswith('.d')]
        else:
            input_dir = os.path.dirname(args.input)
            data = [args.input]
        # print(data)

        # if args.mz_file != '':
        #     df = pd.read_csv(args.mz_file)
        #     mz_list = df['m/z'].to_numpy()
        #     mz_list_round = np.round(mz_list, 4)
        #     print(mz_list)
        #     acc_list = df['Accession'].to_numpy()
        #     # create directory for each accession
        #     for acc in acc_list:
        #         acc_dir = os.path.join(args.output_dir, acc)
        #         if not os.path.exists(acc_dir):
        #             os.mkdir(acc_dir)
        #         for d in data:
        #             acc_fl_dir = os.path.join(acc_dir, d.split('.')[0])
        #             if not os.path.exists(acc_fl_dir):
        #                 os.mkdir(acc_fl_dir)
        if args.mz_file != '':
            df = pd.read_csv(args.mz_file)
            #df = pd.read_csv(args.mz_file, index_col=0)
            if args.num:
                df = df.head(args.num)
            mz_list = df[df.columns[0]].to_numpy()
            #mz_list = np.asarray(list(set(df['m/z'].to_list())))
            #print(mz_list)
            #mz_list_round = np.round(mz_list, 4)
            #print(mz_list)
            # for d in data:
            #     fl_dir = os.path.join(args.output_dir, d.split('.')[0])
            #     if not os.path.exists(fl_dir):
            #         os.mkdir(fl_dir)

        for d in data:
            print("processing file {}".format(d))
            sample_num = os.path.splitext(os.path.basename(d))[0]

            if args.mz:
                plot_ion_image(input_file=os.path.join(input_dir, d), mz=args.mz,
                               output_file=os.path.join(args.output_dir, sample_num + '.' + args.format),
                               tol=args.tol, unit=args.unit, CLAHE=args.CLAHE, contrast_stretch=args.contrast_stretch,
                               lower=args.lower, upper=args.upper, plot=args.plot, cmap=args.cmap,
                               pyimzml=args.pyimzml, remove_isolated_px=args.remove_iso_px)
            else:
                # for i, mz in enumerate(tqdm(mz_list_round)):
                #     save_dir = os.path.join(os.path.join(args.output_dir, sample_num))
                #     plot_ion_image(input_file=os.path.join(args.input_dir, d), mz=mz,
                #                    output_file=os.path.join(save_dir, sample_num + '_' + str(round(mz, 4)).replace('.', '_') + '.' + args.format),
                #                    tol=args.tol, CLAHE=args.CLAHE, contrast_stretch=args.contrast_stretch,
                #                    lower=args.lower, upper=args.upper, plot=args.plot, cmap=args.cmap,
                #                    pyimzml=args.pyimzml, remove_isolated_px=args.remove_iso_px)
                save_dir = os.path.join(args.output_dir, sample_num)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                with multiprocessing.Pool() as pool:
                    # call the function for each item in parallel
                    pool.map(
                        partial(plot_ion_image, input_file=d,
                                output_file='', tol=args.tol, unit=args.unit, CLAHE=args.CLAHE,
                                contrast_stretch=args.contrast_stretch, lower=args.lower, upper=args.upper,
                                plot=args.plot,
                                cmap=args.cmap, pyimzml=args.pyimzml, remove_isolated_px=args.remove_iso_px,
                                output_dir=save_dir), mz_list)
            # else:
            #     for i, mz in enumerate(tqdm(mz_list_round)):
            #         acc = df.loc[df['m/z'] == mz_list[i], 'Accession'].iloc[0]
            #         save_dir = os.path.join(os.path.join(args.output_dir, acc), sample_num)
            #         plot_ion_image(input_file=os.path.join(args.input_dir, d), mz=mz,
            #                        output_file=os.path.join(save_dir, sample_num + '_' + str(round(mz, 4)).replace('.', '_') + '.' + args.format),
            #                        tol=args.tol, CLAHE=args.CLAHE, contrast_stretch=args.contrast_stretch,
            #                        lower=args.lower, upper=args.upper, plot=args.plot, cmap=args.cmap,
            #                        pyimzml=args.pyimzml, remove_isolated_px=args.remove_iso_px)

