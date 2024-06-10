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

warnings.filterwarnings('ignore', module='pyimzml')


def peak_pick(x, y, snr_thr=3, window_size=11, order=3, smooth=1, plot=0):
    # smooth signal
    if smooth == 1:
        smoothed_y = savgol_filter(y, window_size, order)
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

    if plot != 0:
        plt.plot(x, y, 'lightgray', label='profile')
        plt.plot(x, smoothed_y, label='smoothed')
        plt.plot(x_peaks, y_peaks, 'o', label='peaks')
        plt.plot(x_peaks_above_snr, y_peaks_above_snr, 'x', label='peaks above snr')
        plt.title('Peak picking with SNR={} and MAD={}'.format(snr_thr, mad))
        plt.legend()
        plt.show()

    return x_peaks_above_snr, y_peaks_above_snr, smoothed_y


def peak_picking(path, outdir, snr_thr=3, window_size=11, order=3, smooth=1, quant=0, plot=0):
    # pixel-wise peak picking on raw data saved as .d
    if os.path.splitext(path)[1] == '.d':
        # read .d data
        reader = get_reader(path)
        indices = reader.mz_index

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
                x_peaks, y_peaks, y_smoothed = peak_pick(x_profile, y_profile, snr_thr, window_size, order, smooth, plot)

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
            print(low_perc)
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
    args = parser.parse_args()

    if args.out_dir == '':
        args.out_dir = os.path.abspath(os.path.join(os.path.dirname(args.input), 'peakpicking'))
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    peak_picking(args.input, args.out_dir, args.snr, args.window_size, args.order, args.smooth, args.quant, args.plot)








