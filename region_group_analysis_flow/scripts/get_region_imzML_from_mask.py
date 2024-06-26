import argparse
import tifffile
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.morphology import remove_small_objects

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create new imzML with pixels from a binary mask')
    parser.add_argument('imzML_file', type=str, help='imzML file')
    parser.add_argument('bin_img', type=str, help='binary image of region as tiff file')
    parser.add_argument('-result_dir', type=str, help='directory to save result')
    parser.add_argument('-remove_small_objects', type=bool, default=False,
                        help='set to True to remove small objects from binary image')
    parser.add_argument('-plot', type=bool, default=False, help='directory to save result')
    args = parser.parse_args()

    file_name = os.path.basename(args.imzML_file)

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_file), "region")
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)

    file_name = os.path.basename(args.imzML_file).split('.')[0]

    # read data from imzML file
    p = ImzMLParser(args.imzML_file)
    msi_df = utils.get_dataframe_from_imzML(args.imzML_file, multi_index=False)
    file_name = os.path.splitext(os.path.basename(args.imzML_file))[0]

    # read in binary image
    binary = utils.NormalizeData(tifffile.imread(args.bin_img))

    if args.remove_small_objects:
        fg = binary > 0
        cleaned = remove_small_objects(fg, min_size=10) * 1
        binary[cleaned == 0] = 0

    # get pixel indices of binary image
    bin_img_px_idx_np = np.nonzero(binary)
    bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))
    df = pd.DataFrame.from_dict({'x': bin_img_px_idx_np[1], 'y': bin_img_px_idx_np[0]})

    # generate dataframe with MSI data for pixels from binary image
    bin_df = pd.merge(left=df, right=msi_df, on=['x', 'y'])
    mzs = bin_df.iloc[:, 2:].columns.to_numpy()
    spec_df = bin_df.iloc[:, 2:]

    # write imzML file with pixels in binary image
    with ImzMLWriter(os.path.join(args.result_dir, file_name)) as writer:
        for i in tqdm(range(bin_df.shape[0])):
            writer.addSpectrum(mzs, spec_df.iloc[i, :].to_numpy(), (bin_df.iloc[i, 0], bin_df.iloc[i, 1], 0))


    # # read in binary image
    # bin_img = utils.NormalizeData(tifffile.imread(args.bin_img))
    # if args.plot:
    #     plt.imshow(bin_img)
    #     plt.show()
    #
    # # get pixel indices of binary image
    # bin_img_px_idx_np = np.nonzero(bin_img)
    # bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))
    #
    # # write imzML file with pixels in binary image
    # p = ImzMLParser(args.imzML_file)
    # with ImzMLWriter(os.path.join(args.result_dir, file_name)) as writer:
    #     for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
    #         if (x, y) in bin_img_px_idx:
    #             mzs, intensities = p.getspectrum(idx)
    #             writer.addSpectrum(mzs, intensities, (x, y, z))