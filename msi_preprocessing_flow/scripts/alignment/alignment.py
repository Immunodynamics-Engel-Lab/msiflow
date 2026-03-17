import matplotlib.pyplot as plt
import numpy as np
from random import sample
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg.utils import to_mz, to_ppm


### pybasis function from https://bitbucket.org/iAnalytica/basis_pyproc/src/master/basis/preproc/palign.py
def pmatch_nn(refmz, mz, maxshift):

    """
    Performs nearest neighbour matching of mz or css feature vector to the reference one.

    Args:

        refmz: reference mz feature vector.

        mz: feature vector for alignment.

        maxshift: maximum allowed positional shift.

    Returns:

        refmzidcs: matched indices from refmz feature vector

        mzindcs: matached indices from mz feature vector

    """
    refmz = refmz.flatten()
    mz = mz.flatten()
    nvrbls = len(refmz)

    # map each mz to ref mz via interpolation
    mzindcs = np.round(np.interp(mz, refmz, np.arange(0., nvrbls)))
    mzindcs = (mzindcs.astype(int)).flatten()   # array with index of refmz for each mz

    # indices of mz values which are within maxshift tolerance to mapped refmz
    filtindcs = np.asarray(np.nonzero(np.abs(refmz[mzindcs] - mz) <= maxshift))
    filtindcs = (filtindcs.astype(int)).flatten()   # indices of refmz which have mappings within tolerance

    # count how many peaks are mapped to one ref m/z
    refmzidcs = np.unique(mzindcs[filtindcs])

    # return if no matches
    if len(refmzidcs) == 0:
        # print(
        #     "No matches found between reference mz and measured mz within max_shift. "
        #     f"Check your lock masses or increase max_shift={maxshift}."
        # )
        return np.array([], dtype=int), np.array([], dtype=int)

    refmzidcs = np.asarray(refmzidcs)
    mzbins = np.hstack([np.min(refmzidcs) - 0.5, refmzidcs.flatten() + .5])
    freq = np.histogram(mzindcs[filtindcs], bins=mzbins)
    mzrepidcs = refmzidcs[freq[0].astype(int) > 1]  # indices of refmz which has multiple mappings

    # for multiple mz mapping select mz which is closest to cmz
    mzfilt = mzindcs[filtindcs]     # indices of mz which were mapped within tolerance
    mz = mz[filtindcs]              # actual mz values which were mapped within tolerance
    uniqmzidx = (np.ones([1, len(mzfilt)])).flatten()
    for i in mzrepidcs:
        imzdx = ((np.asarray(np.nonzero(i == mzfilt))).astype(int)).flatten()
        minidx = (np.abs(mz[imzdx] - refmz[i])).argmin()
        uniqmzidx[imzdx] = 0.           # set all which are mapped to i to 0
        uniqmzidx[imzdx[minidx]] = 1.   # only set one which is closest to ref to 1

    uniqmzidx = (np.asarray(np.nonzero(uniqmzidx == 1.))).flatten()
    mzindcs = filtindcs[uniqmzidx]

    return refmzidcs, mzindcs


def calibrate_mz(mz, lockmz, mzmaxshift, mzunits='ppm', method='median', min_locks=5):
    """
    Calibrate m/z values using lock masses.

    Parameters
    ----------
    mz : array-like
        Measured m/z values.
    lockmz : array-like
        Reference lock masses.
    mzmaxshift : float
        Maximum allowed deviation for matching.
    mzunits : str, default 'ppm'
        Units for calibration ('ppm' or 'Da').
    method : str, default 'median'
        Calibration method: 'median' (constant shift) or 'linear' (linear regression).
    min_locks : int, default 5
        Minimum number of lock masses that must be matched within mzmaxshift.

    Returns
    -------
    calibrated_mz : ndarray
        Calibrated m/z values (or original if calibration failed).
    median_dev : float
        Median deviation applied (or -999 if calibration failed).
    """
    mz = np.asarray(mz).flatten()
    lockmz = np.asarray(lockmz).flatten()

    # Convert to ppm if needed
    if mzunits == 'ppm':
        mzvals = to_ppm(mz)
        lockvals = to_ppm(lockmz)
    else:
        mzvals = mz
        lockvals = lockmz

    deviations = []

    # Find closest peak for each lock mass
    for lm in lockvals:
        idx = np.argmin(np.abs(mzvals - lm))
        dev = lm - mzvals[idx]
        if np.abs(dev) <= mzmaxshift:
            deviations.append(dev)

    deviations = np.array(deviations)

    # Check if enough lock masses were matched
    if len(deviations) < min_locks:
        return mz, -999  # fail calibration

    # Apply chosen calibration method
    if method == 'median':
        median_dev = np.median(deviations)
        calibrated_mz = mzvals + median_dev
        if mzunits == 'ppm':
            calibrated_mz = to_mz(calibrated_mz)

    elif method == 'linear':
        # Linear regression using all matched lock peaks
        matched_indices = [np.argmin(np.abs(mzvals - lm)) for lm in lockvals if np.abs(lm - mzvals[np.argmin(np.abs(mzvals - lm))]) <= mzmaxshift]
        X_locks = np.vstack([mzvals[matched_indices], np.ones(len(matched_indices))]).T
        Y_locks = deviations
        coef, _, _, _ = np.linalg.lstsq(X_locks, Y_locks, rcond=None)
        a, b = coef
        calibrated_mz = mzvals + (a * mzvals + b)
        if mzunits == 'ppm':
            calibrated_mz = to_mz(calibrated_mz)
        median_dev = np.median(deviations)

    else:
        raise ValueError("method must be 'median' or 'linear'")

    return calibrated_mz, median_dev


def plot_spectra_for_mass(cmz, mzs, intensities, cmz_intensities, mass, pixel_idx, result_dir, fl_name):
    """
    Plots the cmz on the x-axis, the original spectrum, and the aligned spectrum for a given m/z using stem plots.

    Args:
        cmz: Reference common m/z vector.
        mzs: Original m/z values of the spectrum.
        intensities: Original intensities of the spectrum.
        cmz_intensities: Intensities after alignment to cmz.
        mass: Specific m/z value to plot around.
        pixel_idx: Index of the pixel to be plotted (just for reference).
        result_dir: Directory to save the plots.
    """
    plt.figure(figsize=(10, 8))

    # Calculate ppm difference between m/z and ref m/z
    closest_ref_mass_idx = np.argmin(np.abs(cmz - mass))
    closest_ref_mass = cmz[closest_ref_mass_idx]
    closest_meas_mass_idx = np.argmin(np.abs(mzs - closest_ref_mass))
    closest_meas_mass = mzs[closest_meas_mass_idx]
    ppm_diff = (closest_meas_mass - closest_ref_mass) / mass * 1e6  # PPM difference

    # Plot around the mass (e.g., 0.1 Da range around the mass)
    window_range = 0.25
    lower_bound = mass - window_range
    upper_bound = mass + window_range

    # Extract the relevant range from cmz and spectra
    mask = (cmz >= lower_bound) & (cmz <= upper_bound)
    cmz_window = cmz[mask]
    cmz_intensity_window = cmz_intensities[mask]
    mzs_window = mzs[(mzs >= lower_bound) & (mzs <= upper_bound)]
    intensities_window = intensities[(mzs >= lower_bound) & (mzs <= upper_bound)]

    # Check if there is any data to plot
    if mzs_window.size == 0 or intensities_window.size == 0:
        print(f"Warning: No data to plot for m/z {mass} at pixel {pixel_idx}. Skipping plot.")
        return  # Skip this plot if no data in the range

    # Create two subplots: original vs aligned (before and after)
    plt.subplot(2, 1, 1)
    #plt.stem(mzs_window, intensities_window, basefmt=" ", linefmt="b-", markerfmt="bo", label="Original Spectrum")
    #plt.stem(cmz_window, cmz_intensity_window, basefmt=" ", linefmt="g-", markerfmt="gx", label="Aligned Spectrum")
    plt.stem(mzs_window, intensities_window, basefmt=" ", markerfmt=" ", label="Original Spectrum")
    plt.stem(cmz_window, np.zeros_like(cmz_window), basefmt=" ", linefmt="r", markerfmt="rx", label="cmz (Reference)")
    plt.title(f"Spectrum at m/z {mass} (Pixel {pixel_idx}) - Before Alignment\n"
              f"Closest m/z to {closest_ref_mass:.6f} is {closest_meas_mass:.6f} (PPM = {ppm_diff:.2f})")
    plt.xlabel('m/z')
    plt.ylabel('Intensity [a.u.]')
    plt.legend()

    # Plot aligned spectrum
    plt.subplot(2, 1, 2)
    plt.stem(cmz_window, cmz_intensity_window, basefmt=" ", linefmt="g-", markerfmt=" ", label="Aligned Spectrum")
    plt.stem(cmz_window, np.zeros_like(cmz_window), basefmt=" ", linefmt="r", markerfmt="rx", label="cmz (Reference)")
    plt.title(f"Spectrum at m/z {mass} (Pixel {pixel_idx}) - After Alignment")
    plt.xlabel('m/z')
    plt.ylabel('Intensity [a.u.]')
    plt.legend()

    # Set the same x-axis range for both subplots
    plt.subplot(2, 1, 1)  # First subplot
    plt.xlim(lower_bound, upper_bound)  # Set the x-axis limits to match the range
    plt.xticks(np.arange(lower_bound, upper_bound, 0.05))  # Set x-ticks to be consistent (optional)

    plt.subplot(2, 1, 2)  # Second subplot
    plt.xlim(lower_bound, upper_bound)  # Set the x-axis limits to match the range
    plt.xticks(np.arange(lower_bound, upper_bound, 0.05))  # Set x-ticks to be consistent (optional)

    # Set the same y-axis range for both subplots with some margin
    y_max = max(np.max(intensities_window),
                np.max(cmz_intensity_window))  # Get the max intensity value across both spectra
    y_min = 0  # Set minimum y-limit to 0

    # Add a margin (10%) to the y-limits
    y_margin = 0.1 * (y_max - y_min)
    plt.subplot(2, 1, 1)  # First subplot
    plt.ylim(y_min - y_margin, y_max + y_margin)  # Set the y-limits with margin

    plt.subplot(2, 1, 2)  # Second subplot
    plt.ylim(y_min - y_margin, y_max + y_margin)  # Set the y-limits with margin

    # Save the plot
    save_path = os.path.join(result_dir, f"{fl_name}_mass_{mass}_pixel_{pixel_idx}_alignment.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_deviation_heatmap(median_dev_map, out_file, plot=False):
    # Example: median_dev_map as float array
    # - np.nan = non-MSI pixel
    # - -999 = no match
    # - other values = median deviation

    # Mask non-MSI pixels for base heatmap
    base_mask = np.isnan(median_dev_map) | (median_dev_map == -999)
    masked_heatmap = np.ma.masked_array(median_dev_map, mask=base_mask)

    plt.figure(figsize=(12, 12))

    # Base heatmap (viridis)
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')  # non-MSI pixels -> black
    im = plt.imshow(masked_heatmap, cmap=cmap, origin='lower', interpolation='none')

    # Overlay no-match pixels (-999) in red
    no_match_mask = (median_dev_map == -999)
    plt.imshow(np.ma.masked_where(~no_match_mask, no_match_mask),
               cmap=ListedColormap(['red']), origin='lower', interpolation='none')

    plt.colorbar(im, label='Median deviation (ppm)')
    plt.axis('off')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs alignment to reference based on a nearest neighbor approach')
    parser.add_argument('imzML_fl', type=str, help='imzMl file')
    parser.add_argument('-refmz', type=str, default='', help='provide reference spectrum as numpy array to perform alignment')
    parser.add_argument('-calibrate', type=int, default=0, help="set value to perform lock-mass calibration")
    parser.add_argument('-mass_list', type=str, default='', help="comma-separated list of masses for QC and/or calibration")
    parser.add_argument('-result_dir', type=str, default='',
                        help='directory to store result, default \'\' to save results to directory called alignment')
    parser.add_argument('-max_shift', type=float, default=0.05, help='max mass shift in Da/ppm, default=0.05')
    parser.add_argument('-unit', type=str, default='Da', help='unit (either Da or ppm')
    parser.add_argument('-method', type=str, default='median', help='calibration method for computed mass error: either median or linear, default=median')
    parser.add_argument('-min_locks', type=float, default=0, help='min number of lock masses to use for calibration, default=0 automatically sets to number of provided lock masses')
    parser.add_argument('-debug', type=bool, default=False, help='set to True for debugging')
    args = parser.parse_args()

    if args.result_dir == '':
        if args.calibrate > 0:
            args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "calibration")
        else:
            args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "alignment")
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
        qc_dir = None

    # get common m/z vector
    if args.refmz != '':
        # print('performing alignment on reference spectrum')
        cmz = np.load(args.refmz).astype(np.float32)
        if args.unit == 'ppm':
            cmz = to_ppm(cmz)
    elif args.calibrate > 0 and args.mass_list != '':
        # print('performing lock mass calibration')
        cmz = None
        if args.min_locks == 0:
            min_locks = len(mass_list)
        else:
            min_locks = args.min_locks
    else:
        raise ValueError("Either reference m/z or mass list must be provided.")

    # read imzML file
    p = ImzMLParser(args.imzML_fl)

    # Get random 3 pixel indices for QC
    if mass_list != '':
        random_pixel_indices = sample(range(len(p.coordinates)), 3)
    else:
        random_pixel_indices = None

    # img for heatmap with median deviations of calibration
    if args.calibrate > 0 and mass_list:
        max_x = max(coord[0] for coord in p.coordinates)
        max_y = max(coord[1] for coord in p.coordinates)
        median_dev_map = np.full((max_y, max_x), np.nan, dtype=float)


    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML_fl))) as writer:
        for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
            # read spectra
            mzs, intensities = p.getspectrum(idx)
            mzs = mzs.astype(np.float32)

            # align all data to common m/z vector
            if cmz is not None:
                if args.unit == 'ppm':
                    mzs = to_ppm(mzs)
                #print('cmz={}\nmzs{}\nintensities={}'.format(cmz, mzs, intensities))
                cmz_idx, matchmz_idx = pmatch_nn(cmz, mzs, args.max_shift)
                cmz_intensities = np.zeros(cmz.shape)
                cmz_intensities[cmz_idx] = intensities[matchmz_idx]
                writer.addSpectrum(cmz, cmz_intensities, (x, y, z))

                # Plot for the 3 random selected pixels
                if idx in random_pixel_indices:
                    for mass in mass_list:
                        if args.unit == 'ppm':
                            plot_spectra_for_mass(to_mz(cmz), to_mz(mzs), intensities, cmz_intensities, mass, idx,
                                                  qc_dir, os.path.basename(args.imzML_fl).split('.')[0])
                        else:
                            plot_spectra_for_mass(cmz, mzs, intensities, cmz_intensities, mass, idx, qc_dir,
                                                  os.path.basename(args.imzML_fl).split('.')[0])

            # calibrate spectrum based on lock masses
            elif args.mass_list != '' and args.calibrate > 0:
                #print("mzs={}\nmass list={}".format(mzs, mass_list))
                cal_mzs, median_dev = calibrate_mz(mz=mzs, lockmz=np.array(mass_list, dtype=np.float32), mzmaxshift=args.max_shift, mzunits=args.unit, method=args.method, min_locks=min_locks)
                median_dev_map[y - 1, x - 1] = median_dev
                writer.addSpectrum(cal_mzs, intensities, (x, y, z))
            else:
                raise ValueError("No reference spectrum or lock masses provided.")

    # img heatmap of median deviation of calibration
    if args.calibrate > 0:
        plot_deviation_heatmap(median_dev_map, os.path.join(qc_dir, os.path.basename(args.imzML_fl).split('.')[0] + '_median_lock_mass_deviation.png'), args.debug)

