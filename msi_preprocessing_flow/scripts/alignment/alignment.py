import matplotlib.pyplot as plt
import numpy as np
from random import sample
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm import tqdm
import argparse
import os


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

    # Find the index in cmz closest to the mass
    closest_idx = np.argmin(np.abs(cmz - mass))

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
    plt.title(f"Spectrum at m/z {mass} (Pixel {pixel_idx}) - Before Alignment")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs alignment to reference based on a nearest neighbor approach')
    parser.add_argument('imzML_fl', type=str, help='imzMl file')
    parser.add_argument('refmz', type=str, help='reference file as numpy array')
    parser.add_argument('-mass_list', type=str, default='', help="comma-separated list of masses for QC")
    parser.add_argument('-result_dir', type=str, default='',
                        help='directory to store result, default \'\' to save results to directory called alignment')
    parser.add_argument('-max_shift', type=float, default=0.05, help='max mass shift in Da, default=0.05')
    parser.add_argument('-debug', type=bool, default=False, help='set to True for debugging')
    args = parser.parse_args()

    if args.result_dir == '':
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

    all_mzs = []
    all_ints = []

    # get common m/z vector
    cmz = np.load(args.refmz).astype(np.float32)

    # align all data to common m/z vector
    p = ImzMLParser(args.imzML_fl)

    # Get random 3 pixel indices for QC
    if mass_list != '':
        random_pixel_indices = sample(range(len(p.coordinates)), 3)
    else:
        random_pixel_indices = None

    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML_fl))) as writer:
        for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
            mzs, intensities = p.getspectrum(idx)
            mzs = mzs.astype(np.float32)
            #cmz_intensities = get_ints_for_cmz(cmz, mzs, intensities)
            cmz_idx, matchmz_idx = pmatch_nn(cmz, mzs, args.max_shift)
            cmz_intensities = np.zeros(cmz.shape)
            cmz_intensities[cmz_idx] = intensities[matchmz_idx]
            writer.addSpectrum(cmz, cmz_intensities, (x, y, z))

            # Plot for the 3 random selected pixels
            if idx in random_pixel_indices:
                for mass in mass_list:
                    plot_spectra_for_mass(cmz, mzs, intensities, cmz_intensities, mass, idx, qc_dir, os.path.basename(args.imzML_fl).split('.')[0])