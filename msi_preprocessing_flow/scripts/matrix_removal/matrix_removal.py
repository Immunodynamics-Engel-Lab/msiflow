import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import pandas as pd

import tifffile
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_opening, binary_fill_holes
from skimage.morphology import remove_small_objects, octagon

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils
from pkg.plot import get_mz_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove predefined matrix pixels')
    parser.add_argument('imzML_fl', type=str, help='imzML file')
    #parser.add_argument('matrix_pixels', type=str, help='csv file with matrix pixel coordinates')
    parser.add_argument('matrix_img', type=str, help='binary matrix image')
    parser.add_argument('-proc_matrix_img', type=int, default=1, help='set to 1 to postprocess binary matrix image, default=1')
    parser.add_argument('-pixel_removal', type=int, default=1, help='set to 1 to remove matrix/non-tissue pixels, default=1')
    parser.add_argument('-matrix_subtraction', type=int, default=0, help='set to 1 to subtract matrix signals, default=0')
    parser.add_argument('-matrix_peaks_removal', type=int, default=0, help='set to 1 to remove high matrix peaks, default=0')
    parser.add_argument('-num_matrix_peaks', type=int, default=20, help='number of top peaks which should be removed, default=20')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result, default=\'\' will save results in matrix_removal directory')
    parser.add_argument('-qc', type=int, default=1, help='set to 1 for qc output, default=1')
    parser.add_argument('-plot', type=int, default=0, help='set to 1 to show plots, default=0')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "matrix_removal")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    if args.qc == 1:
        qc_path = os.path.join(args.result_dir, 'quality_control')
        if not os.path.exists(qc_path):
            os.mkdir(qc_path)

    if args.proc_matrix_img == 1:
        img = tifffile.imread(args.matrix_img)
        img = img - img.min()
        img = (img / img.max()).astype(int)
        img = img > 0

        cleaned = remove_small_objects(img, min_size=5)
        #matrix_img = binary_dilation(cleaned, structure=np.ones((2, 2)))
        #cleaned=img
        #struct_elem = octagon(3, 1)
        struct_elem = np.ones((2, 2))
        matrix_img = binary_dilation(cleaned, structure=np.ones((5, 5)))
        matrix_img = binary_erosion(matrix_img, structure=struct_elem)
        #matrix_img = binary_fill_holes(matrix_img)
        #matrix_img = binary_opening(img)
        #matrix_img = binary_closing(matrix_img)

        # struct_elem = np.ones((3, 3))
        # struct_elem = np.ones((3, 3))
        # binary opening: 1. dilation 2. erosion
        # matrix_img = binary_dilation(binary_erosion(img, structure=struct_elem), structure=struct_elem)
        # binary closing: 1. erosion 2. dilation
        # matrix_img = binary_erosion(binary_dilation(open_img, structure=struct_elem), structure=struct_elem)

        if args.qc == 1 and args.proc_matrix_img == 1:
            tifffile.imwrite(os.path.join(args.result_dir, os.path.basename(args.imzML_fl).split('.')[0] + '_postproc_matrix_image.tif'),
                             (matrix_img * 255).astype(np.uint8))
    else:
        matrix_img = utils.NormalizeData(tifffile.imread(args.matrix_img))

    if args.plot:
        plt.imshow(matrix_img)
        plt.show()

    p = ImzMLParser(args.imzML_fl)
    #pyx = (p.imzmldict["max count of pixels y"], p.imzmldict["max count of pixels x"])
    df = utils.get_dataframe_from_imzML(args.imzML_fl, multi_index=False)
    mzs = df.columns[2:].to_numpy()

    # get matrix pixels
    # matrix_pixels_df = pd.read_csv(args.matrix_pixels, index_col=0)
    # matrix_pixels = matrix_pixels_df.to_numpy()
    bin_img_px_idx_np = np.nonzero(matrix_img)
    # print('no. of matrix pixels:', np.count_nonzero(matrix_img))
    bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))
    matrix_pixels_df = pd.DataFrame.from_dict({'x': bin_img_px_idx_np[1], 'y': bin_img_px_idx_np[0]})

    # get matrix spectrum
    matrix_df = pd.merge(left=df, right=matrix_pixels_df, on=['x', 'y'])
    matrix_spec = matrix_df.iloc[:, 2:].mean(axis=0)
    matrix_spec_arr = matrix_spec.to_numpy()
    #print(matrix_df)

    # remove rows with matrix pixels from df
    if args.pixel_removal == 1:
        print('no. pixels before matrix removal: ', df.shape[0])
        df_merge = pd.merge(df, matrix_df, how='outer', on=['x', 'y'], indicator=True)
        df = df.loc[df_merge['_merge'] == 'left_only']
        print('no. pixels after matrix removal: ', df.shape[0])

    if args.matrix_subtraction == 1 or args.matrix_peaks_removal == 1:
        if args.matrix_subtraction == 1:
            # subtract matrix spectrum
            df_sub = df.iloc[:, 2:].sub(matrix_spec_arr, axis=1)
            df_sub[df_sub < 0] = 0  # set negative values to 0
        else:
            # get top n peaks
            n = args.num_matrix_peaks
            matrix_mzs_arr = matrix_df.iloc[:, 2:].columns.to_numpy()
            matrix_mz_peak_idx = np.argpartition(matrix_spec_arr, -n)[-n:]
            matrix_mz_peaks = matrix_mzs_arr[matrix_mz_peak_idx]
            df_sub = df.iloc[:, 2:]
            df_sub.loc[:, matrix_mz_peaks] = 0
            #df_sub = df.drop(columns=matrix_mz_peaks.tolist())
        print(df_sub)

        if args.plot == 1 or args.qc == 1:
            sum_spec_df = utils.get_summarized_spectrum(df, method='mean')
            sum_sub = np.mean(df_sub.to_numpy(), axis=0)
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True)  # frameon=False removes frames
            plt.subplots_adjust(hspace=.0)
            ax1.stem(mzs, sum_spec_df.iloc[0, :].to_numpy(), linefmt='#5E4FA2', label='tissue', markerfmt=' ', basefmt=" ",)
            ax1.stem(mzs, matrix_spec.to_numpy().flatten(), markerfmt=' ', basefmt=" ", linefmt='#9F0142', label='matrix')
            ax2.stem(mzs, sum_sub, linefmt='#5E4FA2', label='matrix subtracted',  markerfmt=' ', basefmt=" ")
            ax1.legend()
            ax2.legend()
            plt.xlabel('m/z')
            plt.ylabel('intensities [a.u.]')

            if args.qc == 1:
                plt.savefig(os.path.join(qc_path, os.path.basename(args.imzML_fl).split('.')[0] + '_matrix_sub_spectrum.svg'))

            if args.plot == 1:
                plt.show()
            plt.close()
    else:
        df_sub = df.iloc[:, 2:]
    print(df_sub)
    print(df)

    # write matrix subtracted data
    p = ImzMLParser(args.imzML_fl)
    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML_fl))) as writer:
        for i in tqdm(range(df_sub.shape[0])):
            writer.addSpectrum(df_sub.columns.to_numpy(), df_sub.iloc[i, :].to_numpy(),
                               (df.iloc[i, 0], df.iloc[i, 1], 0))

