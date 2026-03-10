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
import psutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg.utils import to_mz, to_ppm


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


def plot_histo_around_mz(bin_centers, hist, smoothed, ma, mz_center, step_size=0.1, plot=False, out_dir='', dask=0):
    """
    Plot a histogram around a specific m/z value using bin centers.

    Args:
        bin_centers: Array of bin centers (1D)
        hist: Histogram counts corresponding to bin_centers
        smoothed: Smoothed histogram values corresponding to bin_centers
        ma: Indices of maxima in smoothed histogram (relative to full bin_centers)
        mz_center: m/z value to focus on
        step_size: Window around mz_center (±step_size)
        plot: If True, display the plot
        out_dir: Directory to save the plot (optional)
        dask: If 1, arrays are Dask arrays and need .compute()
    """

    # If using Dask arrays, compute
    if dask == 1:
        bin_centers = bin_centers.compute()
        hist = hist.compute()
        smoothed = smoothed.compute()
        ma = ma.compute()

    # Mask bins within ±step_size of mz_center
    bin_mask = (bin_centers >= mz_center - step_size) & (bin_centers <= mz_center + step_size)
    filtered_centers = bin_centers[bin_mask]
    filtered_hist = hist[bin_mask]
    filtered_smoothed = smoothed[bin_mask]

    # Filter maxima to those within the selected range
    # Map original maxima indices (ma) to filtered_centers
    ma_in_range = [i for i, val in enumerate(filtered_centers) if val in bin_centers[ma]]

    # Find the closest maxima to mz_center
    if len(ma_in_range) > 0:
        distances = np.abs(filtered_centers[ma_in_range] - mz_center)
        closest_idx_in_ma = np.argmin(distances)
        closest_value = filtered_centers[ma_in_range[closest_idx_in_ma]]
        abs_diff = np.abs(closest_value - mz_center)
        ppm_diff = (abs_diff / mz_center) * 1e6
    else:
        closest_value = np.nan
        abs_diff = np.nan
        ppm_diff = np.nan

    # Start plotting
    plt.figure(figsize=(8, 5))
    plt.fill_between(filtered_centers, filtered_hist, step='mid', alpha=0.6,
                     color='steelblue', edgecolor='black', linewidth=0.5, label="Histogram")
    plt.plot(filtered_centers, filtered_smoothed, color='red', label="Smoothed curve")

    # Plot maxima
    for i, idx in enumerate(ma_in_range):
        if i == 0:
            plt.plot(filtered_centers[idx], filtered_smoothed[idx], 'go', ms=5, label="Maxima")
        else:
            plt.plot(filtered_centers[idx], filtered_smoothed[idx], 'go', ms=5)

    # Plot closest point to mz_center
    if not np.isnan(closest_value):
        plt.plot(closest_value, filtered_smoothed[ma_in_range[closest_idx_in_ma]],
                 'x', color='orange', ms=5, label="Closest to mass")

    # Labels and title
    plt.ylabel('Rel. frequency')
    plt.xlabel('m/z')
    plt.title(f'Histogram around {mz_center} ± {step_size} m/z\n'
              f'Closest m/z: {closest_value:.6f}, Abs diff: {abs_diff:.6f}, PPM diff: {ppm_diff:.6f}')
    plt.legend(loc='best')

    # Save plot if directory is provided
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'histo_around_{mz_center}.png'))

    if plot:
        plt.show()

    plt.close()


def plot_full_histo(bin_centers, hist, smoothed, ma, plot=False, dask=0):

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


def get_cmz_histo(mz, no_px, mz_res=0.01, px_perc=0.01, plot=False, dask=0, mass_list=None, qc_dir='', unit='ppm'):
    """
    Generate the common m/z vector (cmz) by computing a histogram of m/z values.

    Args:
        mz: m/z values from the spectra.
        no_px: Total number of pixels (for normalization in histogram).
        mz_res: Resolution of m/z bins in Da.
        px_perc: Intensity percentage threshold for peak selection.
        plot: Boolean indicating whether to plot the histograms.
        dask: Whether to use Dask for computation.
        mass_list: List of specific m/z values to highlight.
        qc_dir: Directory for saving QC plots.
        unit: The unit to use for the histogram ('ppm' or 'Da').

    Returns:
        cmz: The generated common m/z vector.
    """
    print("calculating cmz via histogram...")
    start = time.time()

    if unit == 'ppm':
        mz = to_ppm(mz)  # Convert mz to ppm if unit is ppm

    if dask == 1:
        mz_min = da.min(mz) - 5 * mz_res
        mz_max = da.max(mz) + 5 * mz_res
    else:
        mz_min = np.min(mz) - 5 * mz_res
        mz_max = np.max(mz) + 5 * mz_res

    n_bins = int((np.round((mz_max - mz_min) / mz_res) + 1).astype(int))

    if dask == 1:
        hist, bin_edges = da.histogram(mz, bins=n_bins, range=(mz_min, mz_max), weights=da.ones_like(mz) / no_px)
        hist = hist.compute()  # Pull into memory
        bin_edges = bin_edges.compute()  # Pull into memory
    else:
        hist, bin_edges = np.histogram(mz, bins=n_bins, range=(mz_min, mz_max), weights=np.ones_like(mz) / no_px)

    print(f"\nhistogram generated within {time.time() - start} seconds")

    # Smooth the histogram for better peak detection
    smoothed = smooth1D(bin_edges, hist, dask=0)
    print(f"\nsmoothed within {time.time() - start} seconds")

    # Find the peaks in the smoothed histogram
    ma, _ = find_peaks(smoothed, height=None, threshold=None, distance=None, prominence=None, width=None,
                       wlen=None, rel_height=0.5, plateau_size=None)
    print(f"\npeaks found within {time.time() - start} seconds")

    # Apply pixel percentage threshold
    ma = ma[hist[ma] >= px_perc]

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    cmz = bin_centers[ma]

    if unit == 'ppm':
        cmz = to_mz(cmz)
        bin_centers = to_mz(bin_centers)

    if mass_list:
        # plot histogram around specified m/z values
        for mz in mass_list:
            plot_histo_around_mz(bin_centers=bin_centers, hist=hist, smoothed=smoothed, ma=ma, mz_center=mz, plot=plot,
                                 out_dir=qc_dir, dask=0)

    if plot:
        # plot full histogram
        plot_full_histo(bin_centers=bin_centers, hist=hist, smoothed=smoothed, ma=ma, plot=plot, dask=0)

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
    parser.add_argument('-unit', type=str, default='Da', help="mass unit (either Da or ppm)")
    args = parser.parse_args()

    args.px_perc = args.px_perc / 100

    if args.result_dir == '':
        args.result_dir = os.path.join((args.imzML_dir), "alignment")
    os.makedirs(args.result_dir, exist_ok=True)

    if args.mass_list != '':
        try:
            mass_list = [float(x) for x in args.mass_list.split(',')]
        except ValueError:
            print("Error: All elements in mass list must be valid floats.")
        qc_dir = os.path.join(args.result_dir, "quality_control")
        os.makedirs(qc_dir, exist_ok=True)
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
    cmz = get_cmz_histo(mz=all_mzs, no_px=num_pxs, mz_res=args.mz_res, px_perc=args.px_perc, plot=args.debug, dask=args.dask, mass_list=mass_list, qc_dir=qc_dir, unit=args.unit)

    #print('reduced m/z vector from {} to {} bins'.format(np.unique(all_mzs).shape, cmz.shape))
    print('reduced m/z vector to {} bins'.format(cmz.shape[0]))
    np.save(os.path.join(args.result_dir, 'cmz.npy'), cmz)
