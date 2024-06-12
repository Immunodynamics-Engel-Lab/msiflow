import argparse
import tifffile
import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from pyimzml.ImzMLWriter import ImzMLWriter
import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts spectra of specific pixels in binary images')
    parser.add_argument('imzML_file', type=str, help='path to imzML file')
    parser.add_argument('bin_img', help='path to binary image', type=str)
    parser.add_argument('output', type=str, help='output file')
    parser.add_argument('-bin_img2', default='', help='path to second binary image', type=str)
    parser.add_argument('-non_overlap', default=False, help='set to True to use pixels from second binary image'
                                                             'which do not overlap with first binary image')
    parser.add_argument('-exclude_inside_mask', default=False, help='set to True to exclude inside mask from first binary image')
    parser.add_argument('-remove_small_objects', type=bool, default=False,
                        help='set to True to remove small objects from binary image')
    parser.add_argument('-plot', default=False, type=bool, help='set to True to plot images')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    p = ImzMLParser(args.imzML_file)
    pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
    msi_df = utils.get_dataframe_from_imzML(args.imzML_file, multi_index=False)
    fl_name = os.path.basename(args.imzML_file).split('.')[0]
    binary = utils.NormalizeData(tifffile.imread(args.bin_img))

    if args.remove_small_objects:
        fg = binary > 0
        cleaned = remove_small_objects(fg, min_size=10) * 1
        binary[cleaned == 0] = 0

    if args.plot:
        plt.imshow(binary)
        plt.show()

    # get pixel indices of binary image
    if args.bin_img2 != '':
        binary2 = utils.NormalizeData(tifffile.imread(args.bin_img2))
        bin = np.zeros(binary.shape)

        if args.non_overlap:
            bin = np.where(np.logical_and(binary == 0, binary2 == 1), 1, 0)
            if args.exclude_inside_mask:
                inside_mask = np.where(np.logical_and(binary_fill_holes(binary) > 0, binary == 0), 1, 0)
                plt.imshow(inside_mask)
                plt.show()
                bin[inside_mask == 1] = 0
        else:
            bin[(binary == 1) & (binary2 == 1)] = 1
        bin_img_px_idx_np = np.nonzero(bin)
        # print(np.count_nonzero(bin))
        if args.plot:
            plt.imshow(bin)
            plt.show()
    else:
        bin_img_px_idx_np = np.nonzero(binary)
        # print(np.count_nonzero(binary))
    bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))
    df = pd.DataFrame.from_dict({'x': bin_img_px_idx_np[1], 'y': bin_img_px_idx_np[0]})

    # get pixels from binary image
    # get pixel spectrum
    bin_df = pd.merge(left=df, right=msi_df, on=['x', 'y'])
    mzs = bin_df.iloc[:, 2:].columns.to_numpy()
    spec_df = bin_df.iloc[:, 2:]
    # print(mzs)
    # print(bin_df)
    # print(spec_df)
    # print(mzs.shape)
    #
    # print("-------")
    # print(spec_df.iloc[0, :])
    # print(int(bin_df.iloc[0, 0]))
    # print(int(bin_df.iloc[0, 1]))

    # write data of binary pixels
    with ImzMLWriter(args.output) as writer:
        for i in tqdm(range(bin_df.shape[0])):
            writer.addSpectrum(mzs, spec_df.iloc[i, :].to_numpy(), (bin_df.iloc[i, 0], bin_df.iloc[i, 1], 0))
