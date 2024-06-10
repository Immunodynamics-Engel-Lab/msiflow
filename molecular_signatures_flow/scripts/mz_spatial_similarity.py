import argparse
import os
import numpy as np
import pandas as pd
import tifffile
from pyimzml.ImzMLParser import ImzMLParser
import warnings
import sys
from skimage.exposure import rescale_intensity
import time
import multiprocessing
from functools import partial

sys.path.append("..")
from pkg.utils import NormalizeData, get_dataframe_from_imzML, get_similarity_measures, get_mz_img

warnings.filterwarnings('ignore', module='pyimzml')
pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate spatial correlation of all m/z values of imzML file '
                                                 'to img file')
    parser.add_argument('imzML', type=str, help='imzML_file')
    parser.add_argument('img', type=str, help='image to which correlation should be determined')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store results')
    parser.add_argument('-contrast_stretch', type=bool, default=False, help='set to True for contrast stretching')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    fl_name = os.path.splitext(os.path.basename(args.imzML))[0]

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), fl_name + "_leadmass")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # read in data
    img = NormalizeData(tifffile.imread(args.img))
    if args.contrast_stretch:
        p0, p99 = np.percentile(img, (0, 99.9))
        img = rescale_intensity(img, in_range=(p0, p99))
    img = img.ravel()
    p = ImzMLParser(args.imzML)
    pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
    msi_df = get_dataframe_from_imzML(args.imzML, multi_index=True)

    mzs = msi_df.columns.to_numpy()
    corr_df = pd.DataFrame(index=mzs)
    cosine_df = pd.DataFrame(index=mzs)

    # get correlation for each m/z value
    print("calculating similarity measures of {} m/z values of {} to reference image...".format(mzs.shape[0], fl_name))
    start = time.time()
    similarity = []
    # create a process pool that uses all cpus
    with multiprocessing.Pool() as pool:
        # call the function for each item in parallel
        for result in pool.map(partial(get_similarity_measures, pyx=pyx, msi_df=msi_df, img=img,
                                       contrast_stretch=args.contrast_stretch), mzs):
            similarity.append(result)
    pearson, cosine_sim = zip(*similarity)
    print('duration: {}'.format(time.time() - start))

    corr_df[fl_name] = pearson
    cosine_df[fl_name] = cosine_sim

    # sort accordingly
    corr_df = corr_df.sort_values(by=fl_name, ascending=False)
    cosine_df = cosine_df.sort_values(by=fl_name, ascending=False)

    # save spatial correlation as csv file
    corr_df.to_csv(os.path.join(args.result_dir, 'pearson_' + fl_name + '.csv'))
    cosine_df.to_csv(os.path.join(args.result_dir, 'cosine_' + fl_name + '.csv'))

    # save top m/z image
    top_mz = corr_df.index.tolist()[0]
    print("Top m/z value=", top_mz)

    top_mz_img = get_mz_img(pyx, msi_df, top_mz)
    if args.contrast_stretch:
        p0, p99 = np.percentile(top_mz_img, (0, 99.9))
        top_mz_img = rescale_intensity(top_mz_img, in_range=(p0, p99))
    top_mz_img = (NormalizeData(top_mz_img) * 255).astype('uint8')
    tifffile.imwrite(os.path.join(args.result_dir, 'top_mz_' + fl_name + '.tif'), data=top_mz_img)


