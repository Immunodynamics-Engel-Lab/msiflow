from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import os
import random
from scipy.signal import find_peaks
import dask.array as da
from scipy.signal import lfilter
import multiprocessing


class lfilter_dask:
    def __init__(self, b, a):
        self.b = b
        self.a = a

    def compute_lfilter(self, arr):
        return lfilter(b=self.b, a=self.a, x=arr.astype(float))


class findpeaks_dask:
    def __init__(self, height=None, threshold=None, distance=None, prominence = None, width = None,
                 wlen = None, rel_height = 0.5, plateau_size = None):
        self.height = height
        self.threshold = threshold
        self.distance = distance
        self.prominence = prominence
        self.width = width
        self.wlen = wlen
        self.rel_height = rel_height
        self.plateau_size = plateau_size

    def compute_findpeaks(self, arr):
        ma, _ = find_peaks(x=arr.astype(float), height=self.height, threshold=self.threshold, distance=self.distance,
                          prominence=self.prominence, width=self.width, wlen=self.wlen, rel_height=self.rel_height,
                          plateau_size=self.plateau_size)
        return ma


### adapted from pybasis function from https://bitbucket.org/iAnalytica/basis_pyproc/src/master/basis/preproc/palign.py
def get_cmz_histo(mz, no_px, mz_res=0.01, px_perc=0.01, plot=False, dask=0):
    print("calculating cmz via histogram...")
    start = time.time()
    if dask == 1:
        mz_min = da.min(mz) - 5 * mz_res
        mz_max = da.max(mz) + 5 * mz_res
    else:
        mz_min = np.min(mz) - 5 * mz_res
        mz_max = np.max(mz) + 5 * mz_res
    n_bins = int((np.round((mz_max - mz_min) / mz_res) + 1).astype(int))
    if dask == 1:
        hist, bin_edges = da.histogram(mz, bins=n_bins, range=(da.min(mz), da.max(mz)), weights=da.zeros_like(mz) + 1. / no_px)
    else:
        hist, bin_edges = np.histogram(mz, bins=n_bins, weights=np.zeros_like(mz) + 1. / no_px)
    #ma = argrelextrema(hist, np.greater)[0]
    # ma, _ = find_peaks(hist, height=0.01)
    # cmz = bin_edges[ma]
    print("\nhistogram generated within {}".format(time.time() - start))
    smoothed = smooth1D(bin_edges, hist, dask=dask)
    print("\nsmoothed within {}".format(time.time() - start))
    #ma, _ = find_peaks(smoothed, height=0.05)
    if dask == 1:
        findpeaks_func = findpeaks_dask(height=None, threshold=None, distance=None, prominence = None, width = None,
                                        wlen = None, rel_height = 0.5, plateau_size = None)
        ma = da.map_overlap(findpeaks_func.compute_findpeaks, smoothed)
    else:
        ma, _ = find_peaks(smoothed, height=None, threshold=None, distance=None, prominence = None, width = None,
                       wlen = None, rel_height = 0.5, plateau_size = None)
    print("\npeaks found within {}".format(time.time() - start))
    ma = ma[hist[ma] >= px_perc]
    cmz = bin_edges[ma]
    if plot:
        #plt.hist(mz, n_bins, density=True)
        plt.plot(bin_edges[:-1], hist)
        #plt.stem(bin_edges[:-1], hist, markerfmt=' ', basefmt=" ")
        plt.plot(bin_edges[:-1], smoothed, label="smoothed")
        plt.plot(bin_edges[ma], smoothed[ma], 'go', ms=5, label="maxima")
        plt.ylabel('Rel. frequency')
        plt.legend()
        plt.xlabel('Data')
        plt.title('Histogram')
        plt.show()
    return cmz


### function from https://bitbucket.org/iAnalytica/basis_pyproc/src/master/basis/utils/signalproc.py
def smooth1D(x, y, window=10, method='loess', weighting='tri-cubic', dask=0):
    """
    Performs fast smoothing of evenly spaced data using moving loess, lowess or average
    filters.

    References:
        [1] Bowman and Azzalini "Applied Smoothing Techniques for Data Analysis"
        Oxford Science Publications, 1997.

    Args:
        x: Uniformly spaced feature vector (eg mz or drift time).
        y: Array of intensities. Smmothing is computed on flattened array of
            intensities.
        method: Smoothing method {'lowess','loess',or 'average'}, by default 'loess'.
        window: Frame length for sliding window [10 data points, by default].
        weighting: Weighting scheme for smoothing {'tricubic' (default), 'gaussian' or 'linear'}.

    Returns:
        yhat: Smoothed signal.
    """

    from scipy import signal
    from scipy import linalg

    leny = len(y)
    halfw = np.floor((window / 2.))
    window = int(2. * halfw + 1.)
    x1 = np.arange(1. - halfw, (halfw - 1.) + 1)

    if weighting == 'tri-cubic':
        weight = (1. - np.divide(np.abs(x1), halfw) ** 3.) ** 1.5
    elif weighting == 'gaussian':
        weight = np.exp(-(np.divide(x1, halfw) * 2.) ** 2.)
    elif weighting == 'linear':
        weight = 1. - np.divide(np.abs(x1), halfw)

    if method == 'loess':
        V = (np.vstack((np.hstack(weight), np.hstack(weight * x1), np.hstack(weight * x1 * x1)))).transpose()
        order = 2
    elif method == 'lowess':
        V = (np.vstack((np.hstack((weight)), np.hstack((weight * x1))))).transpose()
        order = 1
    elif method == 'average':
        V = weight.transpose()
        order = 0

        # % Do QR decomposition
    [Q, R] = linalg.qr(V, mode='economic')

    halfw = halfw.astype(int)
    alpha = np.dot(Q[halfw - 1,], Q.transpose())

    if dask == 1:
        lfilter_func = lfilter_dask(
            b = alpha * weight,
            a = 1
        )
        yhat = da.map_overlap(lfilter_func.compute_lfilter, y)
    else:
        yhat = signal.lfilter(alpha * weight, 1, y)
    yhat[int(halfw + 1) - 1:-halfw] = yhat[int(window - 1) - 1:-1]

    x1 = np.arange(1., (window - 1.) + 1)
    if method == 'loess':
        V = (np.vstack((np.hstack(np.ones([1, window - 1])), np.hstack(x1), np.hstack(x1 * x1)))).transpose()
    elif method == 'lowess':
        V = (np.vstack((np.hstack(np.ones([1, window - 1])), np.hstack(x1)))).transpose()
    elif method == 'average':
        V = np.ones([window - 1, 1])

    for j in np.arange(1, (halfw) + 1):
        # % Compute weights based on deviations from the jth point,
        if weighting == 'tri-cubic':
            weight = (1. - np.divide(np.abs((np.arange(1, window) - j)), window - j) ** 3.) ** 1.5
        elif weighting == 'gaussian':
            weight = np.exp(-(np.divide(np.abs((np.arange(1, window) - j)), window - j) * 2.) ** 2.)
        elif method == 'linear':
            weight = 1. - np.divide(np.abs(np.arange(1, window) - j), window - j)

        W = (np.kron(np.ones((order + 1, 1)), weight)).transpose();
        [Q, R] = linalg.qr(V * W, mode='economic')

        alpha = np.dot(Q[j - 1,], Q.transpose())
        alpha = alpha * weight
        yhat[int(j) - 1] = np.dot(alpha, y[:int(window) - 1])
        yhat[int(-j)] = np.dot(alpha, y[np.arange(leny - 1, leny - window, -1, dtype=int)])

    return yhat


def get_mzs(imzfile):
    print("reading all m/z values from {}".format(imzfile))
    imzfile = ImzMLParser(imzfile, parse_lib='ElementTree')
    n_intensities = sum(imzfile.intensityLengths[:10])
    num_pxs = len(imzfile.coordinates[:10])
    sp_indcs = np.concatenate((np.array([0]), np.cumsum(imzfile.intensityLengths[:10])))
    mz = da.zeros(n_intensities, chunks='auto')
    for idx, _ in enumerate(tqdm(imzfile.coordinates[:10])):
        imz, _ = imzfile.getspectrum(idx)
        mz[sp_indcs[idx]:sp_indcs[idx + 1]] = imz
    return num_pxs, mz


if __name__ == '__main__':
    # refmz = np.array([0.50, 0.70, 1.50, 1.75, 2.50])
    # mz = np.array([0.40, 0.60, 0.80])
    # mz_ints = np.array([1, 2, 3])
    # maxshift = 0.3
    # cmz, matchmz = pmatch_nn(refmz, mz, maxshift)
    # print(cmz)
    # print(matchmz)
    # cmz_ints = np.zeros(refmz.shape)
    # cmz_ints[cmz] = mz_ints[matchmz]
    # print(cmz_ints)

    # mz = np.array([0.4, 0.45, 0.5, 0.6,
    #                1.0, 1.1, 1.15, 1.15, 1.15,
    #                1.5, 1.55, 1.55,
    #                1.9, 1.95, 2.0, 2.0, 2.0])
    # mzres = 0.1
    # mzmaxshift = 0.5
    # mzunits = 'Da'
    # #refmz = get_reference(mz, mzres, mzmaxshift, mzunits)
    # refmz = get_cmz_histo(mz, 5, mzres, plot=True)
    # print(refmz)

    parser = argparse.ArgumentParser(description='Extracts reference spectrum from multiple MSI data using a kernel '
                                                 'density approach')
    parser.add_argument('imzML_dir', type=str, help='directory with imzML files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result, default=\'\' to create directory called alignment')
    parser.add_argument('-mz_res', type=float, default=0.005, help='expected m/z resolution, default=0.005')
    parser.add_argument('-px_perc', type=float, default=1, help='peak must be in at least this percentage of pixels, default=0.01')
    parser.add_argument('-num_px_perc', type=int, default=100, help='number of pixels in percentage to take from each'
                                                                   'sample to form common m/z vector'
                                                                   '- the higher, the more memory consuming, default=100')
    parser.add_argument('-dask', type=int, default=0, help='set to 1 to use dask, but not yet fully implemented')
    parser.add_argument('-debug', type=bool, default=False, help='set to True for debugging')
    args = parser.parse_args()

    args.px_perc = args.px_perc / 100

    if args.result_dir == '':
        args.result_dir = os.path.join((args.imzML_dir), "alignment")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    imzML_files = [f for f in os.listdir(args.imzML_dir) if os.path.isfile(os.path.join(args.imzML_dir, f))
                   and f.endswith('.imzML') and not f.startswith('.')]
    imzML_paths = [os.path.join(args.imzML_dir, f) for f in imzML_files]

    if args.dask == 1:
        res = []
        with multiprocessing.Pool() as pool:
            for result in pool.map(get_mzs, imzML_paths):
                res.append(result)
        pool.close()
        num_pxs, mz_list = zip(*res)
        num_pxs = np.sum(num_pxs)
        all_mzs = mz_list[0]
        print("merging all mz values into one array")
        for i in tqdm(range(1, len(mz_list))):
            all_mzs = da.concatenate([all_mzs, mz_list[i]], axis=0)
    else:
        all_mzs = []
        num_pxs = 0
        print("reading all m/z values")
        for fl in tqdm(imzML_files):
            p = ImzMLParser(os.path.join(args.imzML_dir, fl))
            # only take specific percentage of pixels randomly
            num_px = int((len(p.coordinates) / 100) * args.num_px_perc)
            num_pxs += num_px
            idx_list = random.sample(range(0, len(p.coordinates)), num_px)
            # num_px = int((len(p.coordinates[:10]) / 100) * args.num_px_perc)
            # num_pxs += len(p.coordinates[:10])
            # idx_list = range(0, len(p.coordinates[:10]))
            for id in idx_list:
                mzs, _ = p.getspectrum(id)
                all_mzs.extend(mzs)
        all_mzs = np.asarray(all_mzs).astype(np.float32)
    #print(all_mzs.shape)
    #print(num_pxs)

    # get common m/z vector
    cmz = get_cmz_histo(mz=all_mzs, no_px=num_pxs, mz_res=args.mz_res, px_perc=args.px_perc, plot=args.debug, dask=args.dask)

    if args.dask == 1:
        cmz = np.array(cmz)

    #print(cmz)

    #print('reduced m/z vector from {} to {} bins'.format(np.unique(all_mzs).shape, cmz.shape))
    print('reduced m/z vector to {} bins'.format(cmz.shape[0]))
    np.save(os.path.join(args.result_dir, 'cmz.npy'), cmz)
