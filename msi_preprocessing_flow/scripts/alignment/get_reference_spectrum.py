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
from concurrent.futures import ProcessPoolExecutor


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


def plot_histo_around_mz(bin_edges, hist, smoothed, ma, mz_center, step_size=0.1, plot=False, out_dir='', dask=0):
    # Find the range of bin edges around mz_center
    bin_mask = (bin_edges[:-1] >= mz_center - step_size) & (bin_edges[1:] <= mz_center + step_size)

    # Extract the relevant data within the selected range
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # If dask is enabled, compute the necessary arrays
    if dask == 1:
        bin_centers = bin_centers.compute()
        hist = hist.compute()
        smoothed = smoothed.compute()
        ma = ma.compute()

    filtered_centers = bin_centers[bin_mask]
    filtered_hist = hist[bin_mask]
    filtered_smoothed = smoothed[bin_mask]
    filtered_ma = [m for m in ma if filtered_centers[0] <= bin_centers[m] <= filtered_centers[-1]]
    reindexed_ma = [np.where(filtered_centers == bin_centers[m])[0][0] for m in filtered_ma]

    # Find the index of the closest value to mz_center using the filtered `ma` indices
    distances = np.abs(filtered_centers[reindexed_ma] - mz_center)  # Calculate absolute distances
    closest_idx_in_ma = np.argmin(distances)  # Get the index of the minimum distance in `ma`

    # Get the corresponding closest bin center from filtered_centers
    closest_value = filtered_centers[reindexed_ma[closest_idx_in_ma]]

    # Calculate the absolute difference and PPM difference
    abs_diff = np.abs(closest_value - mz_center)
    ppm_diff = (abs_diff / mz_center) * 1e6

    # Plotting
    plt.figure(figsize=(8, 5))

    # Plot the filled histogram for the selected range
    plt.fill_between(
        filtered_centers,
        filtered_hist,
        step='mid',
        alpha=0.6,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5,
        label=f"Histogram"
    )

    # Overlay the smoothed curve
    plt.plot(filtered_centers, filtered_smoothed, color='red', label="Smoothed curve")

    # Mark the maxima
    for i, idx in enumerate(reindexed_ma):
        if i == 0:  # Add the label only for the first maxima
            plt.plot(filtered_centers[idx], filtered_smoothed[idx], 'go', ms=5, label="Maxima")
        else:  # For other maxima, do not add the label again
            plt.plot(filtered_centers[idx], filtered_smoothed[idx], 'go', ms=5)

    # Mark the closest point to mz_center
    plt.plot(closest_value, filtered_smoothed[reindexed_ma[closest_idx_in_ma]], 'x', color='orange', ms=5,
             label=f"Closest to mass")

    # Labels and title
    plt.ylabel('Rel. frequency')
    plt.xlabel('m/z')
    plt.title(f'Histogram around {mz_center} ± {step_size} m/z\n'
              f'Closest m/z: {closest_value:.6f}, '
              f'Abs diff: {abs_diff:.6f}, PPM diff: {ppm_diff:.6f}')
    plt.legend(loc='best')

    plt.savefig(os.path.join(out_dir, 'histo_around_{}.png'.format(mz_center)))

    if plot:
        plt.show()


def plot_full_histo(bin_edges, hist, smoothed, ma, plot=False, dask=0):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if dask == 1:
        bin_centers = bin_centers.compute()  # Compute Dask array
        hist = hist.compute()
        smoothed = smoothed.compute()

    plt.figure(figsize=(8, 5))
    plt.fill_between(
        bin_centers,
        hist,
        step='mid',
        alpha=0.6,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5,
        label="histogram"
    )

    # Overlay smoothed curve
    plt.plot(bin_centers, smoothed, color='red', label="smoothed")

    # Mark maxima
    plt.plot(bin_centers[ma], smoothed[ma], 'go', ms=5, label="maxima")

    plt.ylabel('Rel. frequency')
    plt.xlabel('Data')
    plt.title('Histogram')
    plt.legend()

    if plot:
        plt.show()


### adapted from pybasis function from https://bitbucket.org/iAnalytica/basis_pyproc/src/master/basis/preproc/palign.py
def get_cmz_histo(mz, no_px, mz_res=0.01, px_perc=0.01, plot=False, dask=0, mass_list=None, qc_dir=''):
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
    if mass_list:
        # plot histogram around specified m/z values
        for mz in mass_list:
            plot_histo_around_mz(bin_edges=bin_edges, hist=hist, smoothed=smoothed, ma=ma, mz_center=mz, plot=plot, out_dir=qc_dir, dask=dask)
    if plot:
        # plot full histogram
        plot_full_histo(bin_edges=bin_edges, hist=hist, smoothed=smoothed, ma=ma, plot=plot, dask=dask)
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
    n_intensities = sum(imzfile.intensityLengths)
    num_pxs = len(imzfile.coordinates)
    sp_indcs = np.concatenate((np.array([0]), np.cumsum(imzfile.intensityLengths)))
    mz = da.zeros(n_intensities, chunks='auto')
    for idx, _ in enumerate(tqdm(imzfile.coordinates)):
        imz, _ = imzfile.getspectrum(idx)
        mz[sp_indcs[idx]:sp_indcs[idx + 1]] = imz
    return num_pxs, mz


def read_file_numpy(fl, imzml_dir, perc):
    """
    Reads one file and returns:
    - numpy array of mz values (float32)
    - number of sampled pixels
    """
    p = ImzMLParser(os.path.join(imzml_dir, fl))

    n_coords = len(p.coordinates)
    num_px = int(n_coords * perc / 100)

    idx_list = np.random.choice(n_coords, size=num_px, replace=False)

    mzs_list = []
    for idx in idx_list:
        mzs, _ = p.getspectrum(idx)
        mzs_list.append(np.asarray(mzs, dtype=np.float32))

    if len(mzs_list) == 0:
        return np.empty(0, dtype=np.float32), 0

    return np.concatenate(mzs_list), num_px


def collect_all_mzs_dask(imzML_files, imzml_dir, perc, n_jobs=None):
    """
    Returns
    -------
    all_mzs_dask : dask.array.Array  (lazy)
    total_num_pxs : int
    """

    print("reading all m/z values (Dask mode)")

    dask_arrays = []
    total_num_pxs = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(
            tqdm(
                executor.map(
                    read_file_numpy,
                    imzML_files,
                    [imzml_dir] * len(imzML_files),
                    [perc] * len(imzML_files),
                ),
                total=len(imzML_files),
            )
        )

    for mz_array, num_px in results:
        total_num_pxs += num_px

        # Wrap numpy array as a single Dask chunk
        dask_chunk = da.from_array(
            mz_array,
            chunks=len(mz_array)  # one chunk per file
        )

        dask_arrays.append(dask_chunk)

    if len(dask_arrays) == 0:
        return da.from_array(np.empty(0, dtype=np.float32)), 0

    all_mzs_dask = da.concatenate(dask_arrays)

    return all_mzs_dask, total_num_pxs


def collect_all_mzs(imzML_files, imzml_dir, perc, n_jobs=None):
    """
    Returns
    -------
    all_mzs : np.ndarray
    num_pxs : int
    """

    print("reading all m/z values")

    all_arrays = []
    total_num_pxs = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(
            tqdm(
                executor.map(
                    read_file_numpy,
                    imzML_files,
                    [imzml_dir] * len(imzML_files),
                    [perc] * len(imzML_files),
                ),
                total=len(imzML_files),
            )
        )

    for mz_array, num_px in results:
        all_arrays.append(mz_array)
        total_num_pxs += num_px

    all_mzs = np.concatenate(all_arrays)

    return all_mzs, total_num_pxs


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
    parser.add_argument('-mass_list', type=str, default='', help="comma-separated list of masses for QC")
    args = parser.parse_args()

    args.px_perc = args.px_perc / 100

    if args.result_dir == '':
        args.result_dir = os.path.join((args.imzML_dir), "alignment")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    if args.mass_list != '':
        try:
            mass_list = [float(x) for x in args.mass_list.split(',')]
        except ValueError:
            print("Error: All elements in mass list must be valid floats.")
        qc_dir = os.path.join(args.result_dir, "quality_control")
        if not os.path.exists(qc_dir):
            os.mkdir(qc_dir)
    else:
        mass_list = None
        qc_dir = ''

    imzML_files = [f for f in os.listdir(args.imzML_dir) if os.path.isfile(os.path.join(args.imzML_dir, f))
                   and f.endswith('.imzML') and not f.startswith('.')]
    imzML_paths = [os.path.join(args.imzML_dir, f) for f in imzML_files]

    if args.dask == 1:
        all_mzs, num_pxs = collect_all_mzs_dask(
            imzML_files=imzML_files,
            imzml_dir=args.imzML_dir,
            perc=args.num_px_perc,
            n_jobs=None
        )
        #all_mzs = all_mzs_dask.compute()
    else:
        all_mzs, num_pxs = collect_all_mzs(
            imzML_files=imzML_files,
            imzml_dir=args.imzML_dir,
            perc=args.num_px_perc,
            n_jobs=None
        )

    # get common m/z vector
    cmz = get_cmz_histo(mz=all_mzs, no_px=num_pxs, mz_res=args.mz_res, px_perc=args.px_perc, plot=args.debug, dask=args.dask, mass_list=mass_list, qc_dir=qc_dir)

    if args.dask == 1:
        cmz = cmz.compute()

    #print(cmz)

    #print('reduced m/z vector from {} to {} bins'.format(np.unique(all_mzs).shape, cmz.shape))
    print('reduced m/z vector to {} bins'.format(cmz.shape[0]))
    np.save(os.path.join(args.result_dir, 'cmz.npy'), cmz)
