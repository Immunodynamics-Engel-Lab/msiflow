import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
from scipy.signal import find_peaks
from scipy.signal import lfilter
from concurrent.futures import ProcessPoolExecutor
import sys
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg.utils import to_mz, to_ppm


def plot_histo_around_mz(bin_centers, hist, smoothed, ma, mz_center, step_size=0.1, plot=False, out_dir=''):
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
    """
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
        plt.savefig(os.path.join(out_dir, f'histo_around_{mz_center}.svg'))

    if plot:
        plt.show()

    plt.close()


def plot_full_histo(bin_centers, hist, smoothed, ma, plot=False):
    plt.figure(figsize=(8, 5))
    plt.fill_between(bin_centers, hist, step='mid', alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.5, label="histogram")

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


def get_cmz_histo(mz, no_px, mz_res=0.01, px_perc=0.01, plot=False, mass_list=None, qc_dir='', unit='ppm', mz_min_da=None, mz_max_da=None):
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

    if mz_min_da is not None and mz_max_da is not None:
        if unit == 'ppm':
            mz_min = to_ppm(mz_min_da) - 5 * mz_res
            mz_max = to_ppm(mz_max_da) + 5 * mz_res
        else:
            mz_min = mz_min_da - 5 * mz_res
            mz_max = mz_max_da + 5 * mz_res
    else:
        mz_min = np.min(mz) - 5 * mz_res
        mz_max = np.max(mz) + 5 * mz_res

    n_bins = int((np.round((mz_max - mz_min) / mz_res) + 1).astype(int))

    hist, bin_edges = np.histogram(mz, bins=n_bins, range=(mz_min, mz_max), weights=np.ones_like(mz) / no_px)

    print(f"\nhistogram generated within {time.time() - start} seconds")

    # Smooth the histogram for better peak detection
    smoothed = smooth1D(bin_edges, hist)
    print(f"\nsmoothed within {time.time() - start} seconds")

    # Find the peaks in the smoothed histogram
    ma, _ = find_peaks(smoothed, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
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
            plot_histo_around_mz(bin_centers=bin_centers, hist=hist, smoothed=smoothed, ma=ma, mz_center=mz, plot=plot, out_dir=qc_dir)

    if plot:
        plot_full_histo(bin_centers=bin_centers, hist=hist, smoothed=smoothed, ma=ma, plot=plot)

    return cmz


### function from https://bitbucket.org/iAnalytica/basis_pyproc/src/master/basis/utils/signalproc.py
def smooth1D(x, y, window=10, method='loess', weighting='tri-cubic'):
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

    yhat = lfilter(alpha * weight, 1, y)
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

        W = (np.kron(np.ones((order + 1, 1)), weight)).transpose()
        [Q, R] = linalg.qr(V * W, mode='economic')

        alpha = np.dot(Q[j - 1,], Q.transpose())
        alpha = alpha * weight
        yhat[int(j) - 1] = np.dot(alpha, y[:int(window) - 1])
        yhat[int(-j)] = np.dot(alpha, y[np.arange(leny - 1, leny - window, -1, dtype=int)])

    return yhat


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


def collect_all_mzs(imzML_files, imzml_dir, perc, n_jobs=None):
    print("reading all m/z values")

    all_arrays = []
    total_num_pxs = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(read_file_numpy, imzML_files, [imzml_dir] * len(imzML_files), [perc] * len(imzML_files)), total=len(imzML_files), desc="reading files"))

    for mz_array, num_px in results:
        all_arrays.append(mz_array)
        total_num_pxs += num_px

    all_mzs = np.concatenate(all_arrays)

    return all_mzs, total_num_pxs


def get_pixel_indices(imzML_files, imzml_dir, perc, seed=42):
    """Sample pixels once so that all processing modes use the same pixels."""
    rng = np.random.default_rng(seed)

    pixel_indices = {}
    total_num_pxs = 0

    for fl in imzML_files:
        p = ImzMLParser(os.path.join(imzml_dir, fl))

        n_coords = len(p.coordinates)
        num_px = int(n_coords * perc / 100)

        if num_px == 0:
            continue

        idx_list = rng.choice(n_coords, size=num_px, replace=False)
        pixel_indices[fl] = idx_list
        total_num_pxs += num_px

    return pixel_indices, total_num_pxs


def read_histogram_chunk(fl, imzml_dir, idx_list, n_bins, mz_min, mz_max, unit, no_px):
    """Read one pixel chunk and return its normalized histogram."""
    p = ImzMLParser(os.path.join(imzml_dir, fl))

    mzs_list = []

    for idx in idx_list:
        mzs, _ = p.getspectrum(int(idx))
        mzs_list.append(np.asarray(mzs, dtype=np.float32))

    if len(mzs_list) == 0:
        return np.zeros(n_bins, dtype=np.float64)

    mz = np.concatenate(mzs_list)

    if unit == 'ppm':
        mz = to_ppm(mz)

    hist, _ = np.histogram(mz, bins=n_bins, range=(mz_min, mz_max), weights=np.ones_like(mz) / no_px)

    return hist


def histogram_chunk_worker(args):
    return read_histogram_chunk(*args)


def collect_chunked_histogram(pixel_indices, imzml_dir, n_bins, mz_min, mz_max, unit, no_px, n_jobs, pixel_chunk_size):
    tasks = []

    for fl, idx_list in pixel_indices.items():
        for start in range(0, len(idx_list), pixel_chunk_size):
            chunk = idx_list[start:start + pixel_chunk_size]
            tasks.append((fl, imzml_dir, chunk, n_bins, mz_min, mz_max, unit, no_px))

    print(f"number of chunks: {len(tasks)}")
    print(f"pixels per chunk: {pixel_chunk_size}")
    print(f"parallel workers: {n_jobs}")

    hist_total = np.zeros(n_bins, dtype=np.float64)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for chunk_hist in tqdm(executor.map(histogram_chunk_worker, tasks), total=len(tasks), desc="histogram chunks"):
            hist_total += chunk_hist

    return hist_total


def get_cmz_histo_chunked(pixel_indices, imzml_dir, no_px, mz_min_da, mz_max_da, mz_res=0.01, px_perc=0.01, plot=False, mass_list=None, qc_dir='', unit='ppm', n_jobs=1, pixel_chunk_size=1000):
    """Memory-safe histogram calculation using parallel pixel chunks."""
    print("calculating cmz via chunked histogram...")
    start = time.time()

    if unit == 'ppm':
        mz_min = to_ppm(mz_min_da) - 5 * mz_res
        mz_max = to_ppm(mz_max_da) + 5 * mz_res
    else:
        mz_min = mz_min_da - 5 * mz_res
        mz_max = mz_max_da + 5 * mz_res

    n_bins = int(np.round((mz_max - mz_min) / mz_res) + 1)

    print(f"histogram range: {mz_min:.6f} - {mz_max:.6f} {unit}")
    print(f"number of bins: {n_bins}")

    hist = collect_chunked_histogram(pixel_indices=pixel_indices, imzml_dir=imzml_dir, n_bins=n_bins, mz_min=mz_min,
                                     mz_max=mz_max, unit=unit, no_px=no_px, n_jobs=n_jobs, pixel_chunk_size=pixel_chunk_size)

    _, bin_edges = np.histogram(np.array([], dtype=np.float32),bins=n_bins,range=(mz_min, mz_max))

    print(f"\nhistogram generated within {time.time() - start} seconds")

    smoothed = smooth1D(bin_edges, hist)
    print(f"\nsmoothed within {time.time() - start} seconds")

    ma, _ = find_peaks(smoothed, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    print(f"\npeaks found within {time.time() - start} seconds")

    ma = ma[hist[ma] >= px_perc]

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if unit == 'ppm':
        cmz = to_mz(bin_centers[ma])
        bin_centers = to_mz(bin_centers)
    else:
        cmz = bin_centers[ma]

    if mass_list:
        for mz_value in mass_list:
            plot_histo_around_mz(bin_centers=bin_centers, hist=hist, smoothed=smoothed, ma=ma, mz_center=mz_value, plot=plot, out_dir=qc_dir)

    if plot:
        plot_full_histo(bin_centers=bin_centers, hist=hist, smoothed=smoothed, ma=ma, plot=plot)

    return cmz


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts reference spectrum from multiple MSI data using a histogram approach')
    parser.add_argument('imzML_dir', type=str, help='directory with imzML files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result')
    parser.add_argument('-mz_res', type=float, default=0.005, help='expected m/z resolution, default=0.005')
    parser.add_argument('-px_perc', type=float, default=1, help='peak must be in at least this percentage of pixels')
    parser.add_argument('-num_px_perc', type=int, default=100, help='percentage of pixels sampled from each sample')
    parser.add_argument('-mode', type=str, choices=['all', 'chunked'], default='all', help='all = all m/z values in RAM; chunked = memory-safe histogram')
    parser.add_argument('-n_jobs', type=int, default=None, help='number of parallel workers')
    parser.add_argument('-pixel_chunk_size', type=int, default=1000, help='pixels per chunk in chunked mode')
    parser.add_argument('-mz_min_da', type=float, default=None, help='fixed minimum m/z in Da')
    parser.add_argument('-mz_max_da', type=float, default=None, help='fixed maximum m/z in Da')
    parser.add_argument('-debug', action='store_true', help='enable debugging plots')
    parser.add_argument('-mass_list', type=str, default='', help='comma-separated list of masses for QC')
    parser.add_argument('-unit', type=str, default='Da', choices=['Da', 'ppm'], help='mass unit for histogram')

    args = parser.parse_args()

    args.px_perc = args.px_perc / 100

    if args.result_dir == '':
        args.result_dir = os.path.join(args.imzML_dir, "alignment")

    os.makedirs(args.result_dir, exist_ok=True)

    if args.mass_list != '':
        try:
            mass_list = [float(x) for x in args.mass_list.split(',')]
        except ValueError:
            print("Error: All elements in mass list must be valid floats.")
            sys.exit(1)

        qc_dir = os.path.join(args.result_dir, "quality_control")
        os.makedirs(qc_dir, exist_ok=True)
    else:
        mass_list = None
        qc_dir = ''

    imzML_files = [f for f in os.listdir(args.imzML_dir) if os.path.isfile(os.path.join(args.imzML_dir, f)) and f.endswith('.imzML') and not f.startswith('.')]

    if len(imzML_files) == 0:
        raise ValueError(f"No imzML files found in {args.imzML_dir}")

    print(f"found {len(imzML_files)} imzML files")
    print(f"mode: {args.mode}")
    print(f"workers: {args.n_jobs}")

    if args.mode == 'all':

        all_mzs, num_pxs = collect_all_mzs(
            imzML_files=imzML_files,
            imzml_dir=args.imzML_dir,
            perc=args.num_px_perc,
            n_jobs=args.n_jobs
        )

        cmz = get_cmz_histo(
            mz=all_mzs,
            no_px=num_pxs,
            mz_res=args.mz_res,
            px_perc=args.px_perc,
            plot=args.debug,
            mass_list=mass_list,
            qc_dir=qc_dir,
            unit=args.unit,
            mz_min_da=args.mz_min_da,
            mz_max_da=args.mz_max_da
        )

        del all_mzs

    else:

        if args.mz_min_da is None or args.mz_max_da is None:
            raise ValueError("chunked mode requires both -mz_min_da and -mz_max_da")

        if args.mz_min_da >= args.mz_max_da:
            raise ValueError("-mz_min_da must be smaller than -mz_max_da")

        pixel_indices, num_pxs = get_pixel_indices(
            imzML_files=imzML_files,
            imzml_dir=args.imzML_dir,
            perc=args.num_px_perc,
            seed=42
        )

        cmz = get_cmz_histo_chunked(
            pixel_indices=pixel_indices,
            imzml_dir=args.imzML_dir,
            no_px=num_pxs,
            mz_min_da=args.mz_min_da,
            mz_max_da=args.mz_max_da,
            mz_res=args.mz_res,
            px_perc=args.px_perc,
            plot=args.debug,
            mass_list=mass_list,
            qc_dir=qc_dir,
            unit=args.unit,
            n_jobs=args.n_jobs,
            pixel_chunk_size=args.pixel_chunk_size
        )

    print('reduced m/z vector to {} bins'.format(cmz.shape[0]))

    np.save(os.path.join(args.result_dir, 'cmz.npy'), cmz)
