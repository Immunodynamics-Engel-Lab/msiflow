import argparse
import tifffile
import os
import sys
import numpy as np
from skimage.morphology import binary_closing, binary_opening, remove_small_objects, disk, binary_dilation, binary_erosion
from skimage.filters import threshold_otsu, threshold_yen
from skimage.filters import gaussian

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils



def segment(img, sigma, thr_method, min_size, bin_closing_size):
    # normalize data
    img = utils.NormalizeData(img)

    # Gaussian smoothing
    if args.sigma != 0:
        img = gaussian(img, sigma=sigma)
    # low, up = np.percentile(img, (0, 99.9))
    # img = rescale_intensity(img, in_range=(low, up))

    # thresholding
    thr = utils.apply_threshold(img, thr_method)
    img = img > thr

    # remove small objects
    img = remove_small_objects(img, min_size)

    # binary closing
    if bin_closing_size > 0:
        img = binary_closing(img, footprint=np.ones((bin_closing_size, bin_closing_size)))

    img = (img * 255).astype('uint8')

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create binary image')
    parser.add_argument('input', type=str, help='input tif image (stack)')
    parser.add_argument('-output', type=str, default='', help='output file to save segmented image')
    parser.add_argument('-sigma', type=int, default=1, help='sigma for Gaussian smoothing')
    parser.add_argument('-thr_method', type=str, default='otsu', help='threshold algorithm',
                        choices=['otsu', 'yen', 'isodata', 'mean', 'minimum', 'triangle'])
    parser.add_argument('-min_size', type=int, default=10, help='all objects below this size will be removed')
    parser.add_argument('-bin_closing_size', type=int, default=0, help='set size of structuring element, '
                                                                       '0 for no binary closing')
    parser.add_argument('-chan_to_segment', type=int, default=10, help='all objects below this size will be removed')
    parser.add_argument('-chan_to_seg_list', type=lambda s: [int(item)-1 for item in s.split(',')], default=[],
                        help='pass delimited list of image channels to segment')
    args = parser.parse_args()

    if args.output == '':
        out_dir = os.path.abspath(os.path.join(os.path.dirname(args.input), 'segmented'))
    else:
        out_dir = os.path.dirname(args.output)
    fn = os.path.basename(args.input)
    args.output = os.path.join(out_dir, fn)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # read data
    img = tifffile.imread(args.input)

    # segment image stack
    if img.squeeze().ndim > 2:
        binary = np.zeros(img.shape)
        for i in range(img.shape[0]):
            if i in args.chan_to_seg_list:
                binary[i] = segment(img[i], args.sigma, args.thr_method, args.min_size, args.bin_closing_size)
            else:
                binary[i] = img[i]
    # segment single image
    else:
        binary = segment(img, args.sigma, args.thr_method, args.min_size, args.bin_closing_size)

    # write segmented output
    tifffile.imwrite(args.output, binary.astype('uint8'), photometric='minisblack')
