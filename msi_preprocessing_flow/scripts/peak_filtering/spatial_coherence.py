import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tifffile
import os
import argparse
from skimage import measure
import numba
from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd
from skimage.exposure import rescale_intensity
import time
import multiprocessing
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils
from pkg.plot import plot_ion_image


def properties_largest_area_cc(ccs):
    """
    Extracts the connected component
    with the largest area.

    Parameters
    ----------
    ccs: numpy.ndarray
        connected components

    Returns
    ----------
    RegionProperties
        connected component with largest area

    """
    regionprops = measure.regionprops(ccs)
    if len(regionprops) == 0:
        return -1
    areas = lambda r: r.area
    argmax = max(regionprops, key=areas)
    return argmax


def spatial_coherence(image):
    """
    Spatial coherence of a binary image,
    that is to say the area of the largest
    connected component.

    Parameters
    ----------
    image: np.ndarrau
        binarized image

    Returns
    ----------
    float
        spatial coherence

    """
    labels = measure.label(image, background=0)
    r = properties_largest_area_cc(labels)
    if r == -1:
        return -1
    else:
        return r.area


def find_min_spatial_coherence(image2D, quantiles=None, upper=100):
    """
    Finds images with spatial
    coherence values greater than a given threshold.

    Spatial coherence values are computed
    for several quantile thresholds. The minimum area
    over the thresholded images is kept.

    Parameters
    ----------
    image2D: np.ndarray
        MALDI image
    quantiles: list
        quantile threshold values (list of integers)
    upper: int
        quantile upper threshold

    Returns
    ----------
    int
        min spatial coherence of different thresholds
    """

    if quantiles is None:
        quantiles = [60, 70, 80, 90]
    norm_img = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
    min_area = sys.maxsize
    upper_threshold = np.percentile(norm_img, upper)
    # plt.imshow(norm_img, cmap='gray')
    # plt.show()
    for quantile in quantiles:
        threshold = int(np.percentile(norm_img, quantile))
        sc = spatial_coherence((norm_img > threshold) & (norm_img <= upper_threshold))
        if sc < min_area:
            min_area = sc
        # fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all')
        # axes[0].imshow(norm_img > threshold, cmap='gray')
        # axes[1].imshow(norm_img <= upper_threshold, cmap='gray')
        # fig.tight_layout()
        # plt.show()
    return min_area


def get_mz_img(pyx, msi_df, mz):
    coords = msi_df.index.tolist()
    msi_img = np.zeros(pyx)
    for x_val, y_val in coords:
        msi_img[y_val, x_val] = msi_df.loc[(x_val, y_val), mz]
    return msi_img


def get_sc(mz):
    mz_img = get_mz_img(pyx, msi_df, mz)
    if args.contrast_stretch:
        # mz_img = gaussian_filter(mz_img, sigma=1)
        p0, p99 = np.percentile(mz_img, (1, 99))
        mz_img = rescale_intensity(mz_img, in_range=(p0, p99))
    mz_img = np.nan_to_num(mz_img)
    # print(np.max(mz_img))
    # plt.imshow(mz_img)
    # plt.show()
    sc = find_min_spatial_coherence(mz_img, quantiles=[60, 70, 80, 90])
    return sc


#msi_img = '/home/phispa/cerebellum/peakpicking/alignment/deisotoping/intranorm_median/P04370/20221208_Brain_Trypsin_2h/20221208_Brain_Trypsin_2h_726_4046.tif'
#msi_img = '/home/phispa/cerebellum/peakpicking/alignment/deisotoping/intranorm_median/P04370/20221208_Brain_Trypsin_2h/20221208_Brain_Trypsin_2h_1237_6436.tif'
#msi_img = '/home/phispa/UPEC/MSI/798_54mz/15.tif'
#find_min_spatial_coherence(msi_img, factor=0, quantiles=[60, 70, 80, 90], upper=100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate spatial correlation of all m/z values of multiple imzML'
                                                 'files')
    parser.add_argument('imzML', type=str, help='imzML_file')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store results, default=\'\' in input directory')
    parser.add_argument('-contrast_stretch', type=bool, default=False, help='set to True for contrast stretching')
    parser.add_argument('-tol', default=0.0001, type=float, help='tolerance')
    parser.add_argument('-CLAHE', default=False, type=bool, help='set to True for CLAHE')
    parser.add_argument('-lower', default=1, type=int, help='lower percentile for contrast stretching')
    parser.add_argument('-upper', default=99, type=int, help='upper percentile for contrast stretching')
    parser.add_argument('-cmap', default='viridis', type=str, help='colormap')
    parser.add_argument('-pyimzml', default=False, type=bool, help='if True, use pyimzml library')
    parser.add_argument('-remove_iso_px', default=False, type=bool, help='if True, isolated pixels are removed '
                                                                         'from image')
    parser.add_argument('-format', default='png', type=str, help='output file format, either png or tif')
    parser.add_argument('-save_imgs', default=False, type=bool, help='set to True to save least and top sc images')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML))
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # read in data
    sample_num = os.path.basename(args.imzML).split('.')[0]
    p = ImzMLParser(args.imzML)
    pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
    #msi_df = utils.get_dataframe_from_imzML(args.imzML, multi_index=True).iloc[:, :20]
    msi_df = utils.get_dataframe_from_imzML(args.imzML, multi_index=True)

    #smallest_diff, smallest_diff_ind, val = utils.find_nearest_value(768.59, msi_df.columns.to_numpy())
    #print(val)
    #mzs = np.asarray([val])
    mzs = msi_df.columns.to_numpy()
    result_df = pd.DataFrame(index=mzs)

    print("calculating spatial coherence of {} m/z values...".format(mzs.shape[0]))
    start = time.time()
    sc = []

    with multiprocessing.Pool() as pool:
        # call the function for each item in parallel
        for result in pool.map(get_sc, mzs):
            sc.append(result)
    print('duration: {}'.format(time.time() - start))
    # for mz in mzs:
    #     result = get_sc(mz)
    #     sc.append(result)

    res_df = pd.DataFrame(index=mzs, columns=['Spatial coherence'], data=sc)
    res_df.to_csv(os.path.join(args.result_dir, sample_num + '_sc.csv'))
    # print(res_df)

    # distribution of spatial coherence of all m/z
    sc = np.asarray(sc)
    step = 500
    sc_min = np.min(sc) - 5 * step
    sc_max = np.max(sc) + 5 * step
    n_bins = int((np.round((sc_max - sc_min) / step) + 1).astype(int))
    hist, bin_edges = np.histogram(sc, n_bins)
    counts, bins = np.histogram(sc, bins=n_bins)

    plt.hist(x=bins[:-1], bins=bins, weights=counts)
    plt.xlabel('spatial coherence')
    plt.ylabel('count')
    plt.savefig(os.path.join(args.result_dir, sample_num + '_sc_distribution.svg'))
    if args.plot:
        plt.show()


    # print("saving top 20 SC images...")
    # for mz in top20_mzs:
    #     plot_ion_image(input_file=args.imzML, mz=mz,
    #                    output_dir=top_dir,
    #                    tol=args.tol, CLAHE=args.CLAHE, contrast_stretch=args.contrast_stretch,
    #                    lower=args.lower, upper=args.upper, plot=args.plot, cmap=args.cmap,
    #                    pyimzml=args.pyimzml, remove_isolated_px=args.remove_iso_px)
    #
    # print("saving bottom 20 SC images...")
    # for mz in least20_mzs:
    #     plot_ion_image(input_file=args.imzML, mz=mz,
    #                      output_dir=bottom_dir,
    #                      tol=args.tol, CLAHE=args.CLAHE, contrast_stretch=args.contrast_stretch,
    #                      lower=args.lower, upper=args.upper, plot=args.plot, cmap=args.cmap,
    #                      pyimzml=args.pyimzml, remove_isolated_px=args.remove_iso_px)

    if args.save_imgs:
        # save ion images with top and least 20% SC
        perc20 = np.percentile(sc, 0.01)
        perc80 = np.percentile(sc, 99.9)
        least20_mzs = mzs[sc <= perc20]
        top20_mzs = mzs[sc >= perc80]
        # print(perc20)
        # print(perc80)
        # print(least20_mzs)
        # print(top20_mzs)

        bottom_dir = os.path.join(args.result_dir, 'bottomSC')
        top_dir = os.path.join(args.result_dir, 'topSC')
        if not os.path.exists(bottom_dir):
            os.mkdir(bottom_dir)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        # print(bottom_dir)
        # print(top_dir)
        with multiprocessing.Pool() as pool:
            # call the function for each item in parallel
            pool.map(partial(plot_ion_image, input_file=args.imzML, output_file='', tol=args.tol, CLAHE=args.CLAHE,
                             contrast_stretch=args.contrast_stretch, lower=args.lower, upper=args.upper, plot=args.plot,
                             cmap=args.cmap, pyimzml=args.pyimzml, remove_isolated_px=args.remove_iso_px,
                             output_dir=bottom_dir), least20_mzs)
            pool.map(partial(plot_ion_image, input_file=args.imzML, output_file='', tol=args.tol, CLAHE=args.CLAHE,
                             contrast_stretch=args.contrast_stretch, lower=args.lower, upper=args.upper, plot=args.plot,
                             cmap=args.cmap, pyimzml=args.pyimzml, remove_isolated_px=args.remove_iso_px,
                             output_dir=top_dir), top20_mzs)



