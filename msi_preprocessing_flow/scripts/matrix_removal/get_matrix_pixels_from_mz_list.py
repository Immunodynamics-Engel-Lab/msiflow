import argparse
import tifffile
import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from pyimzml.ImzMLWriter import ImzMLWriter
from skimage.exposure import equalize_adapthist
import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.morphology import binary_closing, disk

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils
from pkg.plot import get_mz_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates binary image from m/z images')
    parser.add_argument('imzML_file', type=str, help='imzML file')
    parser.add_argument('mzlist', help='m/z list', type=str)
    parser.add_argument('-tol', type=float, default=0.01, help='tolerance for ion image generation')
    parser.add_argument('-threshold', type=str, default='otsu', help='thresholding algorithm',
                        choices=['otsu', 'yen', 'isodata', 'mean', 'minimum', 'triangle'])
    parser.add_argument('-result_dir', type=str, default='', help='directory to save results')
    parser.add_argument('-method', type=str, default='mean', help='method to get combined image')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_file), "matrix_removal")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    mz_list = [float(item) for item in args.mzlist.split(',')]
    p = ImzMLParser(args.imzML_file)
    pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
    msi_df = utils.get_dataframe_from_imzML(args.imzML_file, multi_index=True)
    fl_name = os.path.basename(args.imzML_file).split('.')[0]

    imgs = []
    for mz in mz_list:
        #ion_img = utils.NormalizeData(getionimage(p, mz, tol=0.1))
        ion_img = get_mz_img(pyx, msi_df, mz, tol=args.tol)
        imgs.append(ion_img)
    stacked = np.dstack(imgs)
    mean_img = stacked.sum(axis=2)
    tifffile.imwrite(os.path.join(args.result_dir, fl_name + '_sum_matrix_mzs_img.tif'),
                     data=(utils.NormalizeData(mean_img) * 255).astype('uint8'))

    thresh = utils.apply_threshold(mean_img, args.threshold)
    binary = (mean_img > thresh)
    binary = binary_closing(binary, disk(2)) * 1
    tifffile.imwrite(os.path.join(args.result_dir, fl_name + '_matrix_cluster.tif'),
                    data=(utils.NormalizeData(binary) * 255).astype('uint8'))

    # get pixel indices of binary image
    bin_img_px_idx_np = np.nonzero(binary)
    bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))
    df = pd.DataFrame.from_dict({'x': bin_img_px_idx_np[1], 'y': bin_img_px_idx_np[0]})
    df.to_csv(os.path.join(args.result_dir, fl_name + '_matrix_pixels.csv'), index=True)

# with ImzMLWriter(args.output_file) as writer:
#     for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
#         if (x, y) in bin_img_px_idx:
#             mzs, intensities = p.getspectrum(idx)
#             writer.addSpectrum(mzs, intensities, (x, y, z))


# msi_df = utils.get_dataframe_from_imzML(args.imzML_file, multi_index=True)
# actual_mzs = msi_df.columns.to_numpy()
#
# mz_list = [float(item) for item in args.mzlist.split(',')]
# mzs = []
# for m in mz_list:
#     _, _, val = utils.find_nearest_value(m, actual_mzs)
#     mzs.append(val)
#
# print(actual_mzs)
# print(mzs)
#
# p = ImzMLParser(args.imzML_file)
# pyx = (p.imzmldict["max count of pixels y"], p.imzmldict["max count of pixels x"])
#
# # get combined image of m/z values
# #mz_combi_img = get_combi_mz_img(pyx, msi_df, mzs, method=args.method)
# mz_combi_img = get_mz_img(pyx, msi_df, mzs[0])
# mz_combi_img = (utils.NormalizeData(mz_combi_img) * 255).astype('uint8')
# tifffile.imsave(args.output_file, data=mz_combi_img)

#
# # get pixel indices of binary image
# bin_img_px_idx_np = np.nonzero(bin_img)
# bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))
#
# # write imzML file with pixels in binary image
# p = ImzMLParser(args.imzML_file)
# with ImzMLWriter(args.output_file) as writer:
#     for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
#         if (x, y) in bin_img_px_idx:
#             mzs, intensities = p.getspectrum(idx)
#             writer.addSpectrum(mzs, intensities, (x, y, z))
