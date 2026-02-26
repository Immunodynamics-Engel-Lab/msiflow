from tqdm.auto import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from scipy.signal import savgol_filter
import argparse
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import warnings
import os
import pandas as pd
from imzy import get_reader
import random

warnings.filterwarnings('ignore', module='pyimzml')


def plot_peak_picking_for_mz(x, y, x_peaks, y_peaks, smoothed_y, x_peaks_above_snr, y_peaks_above_snr, snr_thr, mad,
                             mz_center, step_size, qc_dir, plot, sample_name, pixel_id):
    # Filter data to only include points within mz_center ± 0.1
    mask = (x >= mz_center - step_size) & (x <= mz_center + step_size)
    x_filtered = x[mask]
    y_filtered = y[mask]
    smoothed_y_filtered = smoothed_y[mask]

    # Filter peaks within the same range
    peaks_mask = (x_peaks >= mz_center - step_size) & (x_peaks <= mz_center + step_size)
    x_peaks_filtered = x_peaks[peaks_mask]
    y_peaks_filtered = y_peaks[peaks_mask]

    # Filter peaks above SNR within the same range
    peaks_above_snr_mask = (x_peaks_above_snr >= mz_center - step_size) & (
            x_peaks_above_snr <= mz_center + step_size)
    x_peaks_above_snr_filtered = x_peaks_above_snr[peaks_above_snr_mask]
    y_peaks_above_snr_filtered = y_peaks_above_snr[peaks_above_snr_mask]

    # Plot the filtered data
    plt.figure(figsize=(8, 5))
    plt.plot(x_filtered, y_filtered, 'lightgray', label='profile')
    plt.vlines(x_filtered, ymin=0, ymax=y_filtered, colors='lightgray', linewidth=1)
    plt.plot(x_filtered, smoothed_y_filtered, 'red', label='smoothed')
    plt.plot(x_peaks_filtered, y_peaks_filtered, 'o', color='gray', label='peaks', ms=5)
    plt.plot(x_peaks_above_snr_filtered, y_peaks_above_snr_filtered, 'go', label='peaks above snr', ms=5)

    if x_peaks_above_snr_filtered.any():
        # Find the closest m/z value within the filtered range
        distances = np.abs(x_peaks_above_snr_filtered - mz_center)  # Calculate absolute distances from mz_center
        closest_idx = np.argmin(distances)  # Index of the minimum distance (closest value)
        closest_mz = x_peaks_above_snr_filtered[closest_idx]  # The closest m/z value
        abs_deviation = np.abs(closest_mz - mz_center)  # Absolute deviation
        ppm_deviation = (abs_deviation / mz_center) * 1e6  # PPM deviation

        # Mark the closest m/z value with an orange cross
        plt.plot(closest_mz, y_peaks_above_snr_filtered[closest_idx], 'x', color='orange', ms=5,
                 label=f'Closest to mass')

        plt.title(f'Spectra around {mz_center} ± {step_size} m/z '
                  f'(SNR: {snr_thr:.1f}, MAD: {mad:.6f})\n'
                  f'Closest m/z: {closest_mz:.6f}, '
                  f'Abs diff: {abs_deviation:.6f}, PPM diff: {ppm_deviation:.6f}')
    else:
        plt.title(f'Spectra around {mz_center} ± {step_size} m/z '
                  f'(SNR: {snr_thr:.1f}, MAD: {mad:.6f})\n')

    plt.ylabel('Intensity [a.u.]')
    plt.xlabel('m/z')
    plt.legend()

    plt.savefig(os.path.join(qc_dir, '{}_{}_{}_mz.png'.format(sample_name, pixel_id, mz_center)))

    if plot != 0:
        plt.show()


def peak_pick(x, y, snr_thr=3, window_size=11, order=3, smooth=1, plot=0, sample_name=None, px_id=None, mass_list=None, qc_dir=None):
    # smooth signal
    if smooth == 1:
        smoothed_y = savgol_filter(y, window_size, order)
        smoothed_y[smoothed_y < 0] = 0
    else:
        smoothed_y = y

    # find peaks
    peaks, _ = find_peaks(smoothed_y)
    x_peaks = x[peaks]
    y_peaks = smoothed_y[peaks]

    # filter peaks based on SNR where noise is MAD
    mad = np.median(np.absolute(y_peaks - np.median(y_peaks)))
    snr = y_peaks / mad
    idx_above_snr_thr = snr >= snr_thr

    y_peaks_above_snr = y_peaks[idx_above_snr_thr]
    x_peaks_above_snr = x_peaks[idx_above_snr_thr]

    if mass_list and qc_dir:
        step_size = 0.1
        for mz_center in mass_list:
            plot_peak_picking_for_mz(x, y, x_peaks, y_peaks, smoothed_y, x_peaks_above_snr, y_peaks_above_snr, snr_thr,
                                     mad, mz_center, step_size, qc_dir, plot, sample_name, px_id)

    if plot != 0:
        plt.plot(x, y, 'lightgray', label='profile')
        plt.plot(x, smoothed_y, label='smoothed')
        plt.plot(x_peaks, y_peaks, 'o', label='peaks')
        plt.plot(x_peaks_above_snr, y_peaks_above_snr, 'x', label='peaks above snr')
        plt.title('Peak picking with SNR={} and MAD={}'.format(snr_thr, mad))
        plt.legend()
        plt.show()

    return x_peaks_above_snr, y_peaks_above_snr, smoothed_y


def peak_picking(path, outdir, snr_thr=3, window_size=11, order=3, smooth=1, quant=0, plot=0, mass_list=None, qc_dir=None):
    # pixel-wise peak picking on raw data saved as .d
    if os.path.splitext(path)[1] == '.d':
        # read .d data
        reader = get_reader(path)
        indices = reader.mz_index

        # pixel indices for QC
        if mass_list and qc_dir:
            qc_idx = random.sample(range(1, reader.get_n_pixels()), 3)
        else:
            qc_idx = []

        # write peak picked pixel spectrum
        with ImzMLWriter(os.path.join(outdir, os.path.basename(path).split('.')[0] + '.imzML')) as writer:

            # perform pixel-wise peak picking
            for i, frame_id in enumerate(trange(1, reader.get_n_pixels(), desc="Extracting peak...", miniters=50)):
                # read in pixel spectrum
                x_profile = reader.index_to_mz(frame_id, indices)
                y_profile = reader.read_profile_spectrum(frame_id)
                #indices_centroid, y_centroid = reader.read_centroid_spectrum(frame_id)
                #x_centroid = reader.index_to_mz(1, indices_centroid)

                # perform peak picking on pixel spectrum
                if frame_id in qc_idx or plot != 0:
                    x_peaks, y_peaks, y_smoothed = peak_pick(x_profile, y_profile, snr_thr, window_size, order, smooth,
                                                             plot, os.path.basename(path).split('.')[0], frame_id, mass_list, qc_dir)
                else:
                    x_peaks, y_peaks, y_smoothed = peak_pick(x_profile, y_profile, snr_thr, window_size, order, smooth,
                                                             plot, mass_list=None, qc_dir=None)

                writer.addSpectrum(x_peaks, y_peaks, (reader._xyz_coordinates[i][0], reader._xyz_coordinates[i][1], reader._xyz_coordinates[i][2]))

    # peak picking on summerized spectrum saved as .csv file
    elif os.path.splitext(path)[1] == '.csv':
        # read summerized spectrum from .csv file
        df = pd.read_csv(path, index_col=0)
        x = df.index.to_numpy().astype(np.float32)
        y = df[df.columns[0]].to_numpy().astype(np.float32)

        # perform peak picking based on a predefined lower quantile
        if quant != 0:
            low_perc = np.percentile(y, quant)
            # print(low_perc)
            x_peaks = x[y > low_perc]
            y_peaks = y[y > low_perc]
        else:
            print("CAUTION: define a lower quantile of peaks which should be discarded to perform peak picking.")

        if plot != 0:
            plt.plot(x, y)
            plt.plot(x_peaks, y_peaks, marker="o", ls="", ms=3)
            plt.savefig(os.path.join(outdir, os.path.basename(path).split('.')[0] + '.svg'))
            plt.show()

        # save peak picked data as csv file
        result_df = pd.DataFrame.from_dict({'m/z': x_peaks, 'intensity': y_peaks})
        result_df.to_csv(os.path.join(outdir, os.path.basename(path).split('.')[0] + '.csv'))

    # pixel-wise peak picking on raw data saved as .imzML
    else:
        # read imzML data
        p = ImzMLParser(path)

        # write peak picked pixel spectrum
        with ImzMLWriter(os.path.join(outdir, os.path.basename(path).split('.')[0] + '.imzML')) as writer:

            # perform pixel-wise peak picking
            for idx, (x_coord, y_coord, z_coord) in enumerate(tqdm(p.coordinates)):
                # read in pixel spectrum
                x, y = p.getspectrum(idx)

                # perform peak picking on pixel spectrum
                x_peaks, y_peaks, y_smoothed = peak_pick(x, y, snr_thr, window_size, order, smooth, plot)

                writer.addSpectrum(x_peaks, y_peaks, (x_coord, y_coord, z_coord))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes peak picking on a MSI dataset')
    parser.add_argument('input', type=str, help='path to raw .d timsTOF or imzML data for pixel-wise peak picking'
                                                'or path to .csv for peak picking on average spectrum based on intensity quatile')
    parser.add_argument('-out_dir', type=str, default='', help='output directory, default=\'\' to create directory called peakpicking')
    parser.add_argument('-quant', type=float, default=0, help='set a value to discard this lowest percentage of peaks in average spectrum, default=0')
    parser.add_argument('-snr', type=float, default=3, help='SNR threshold, default=3')
    parser.add_argument('-smooth', type=int, default=1, help='set to 1 to perform smoothing, default=1')
    parser.add_argument('-window_size', type=int, default=11, help='window size of Savgol filter, default=11')
    parser.add_argument('-order', type=int, default=3, help='polynomial order of Savgol filter, default=3')
    parser.add_argument('-plot', type=int, default=0, help='set to 1 for plots, default=0')
    parser.add_argument('-mass_list', type=str, default='', help="comma-separated list of masses for QC")
    args = parser.parse_args()

    if args.out_dir == '':
        args.out_dir = os.path.abspath(os.path.join(os.path.dirname(args.input), 'peakpicking'))
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if args.mass_list != '':
        try:
            mass_list = [float(x) for x in args.mass_list.split(',')]
        except ValueError:
            print("Error: All elements in mass list must be valid floats.")
        qc_dir = os.path.join(args.out_dir, "quality_control")
        if not os.path.exists(qc_dir):
            os.mkdir(qc_dir)
    else:
        mass_list = None
        qc_dir = None

    peak_picking(args.input, args.out_dir, args.snr, args.window_size, args.order, args.smooth, args.quant, args.plot, mass_list, qc_dir)








