import argparse
import os
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from msi_preprocessing_flow.scripts.alignment.alignment import pmatch_nn
from pkg.utils import to_ppm


def find_smallest_diff(mzs, mass, unit='Da'):
    """
    Find the nearest m/z in mzs to a given mass and return the difference
    in either Da or ppm.

    Args:
        mzs (np.ndarray): Sorted array of measured m/z values.
        mass (float): Reference mass to compare.
        unit (str): 'Da' for Dalton or 'ppm' for parts-per-million.

    Returns:
        float: Smallest difference in chosen unit.
    """

    if unit == 'ppm':
        mzs_val = to_ppm(mzs)
        mass_val = to_ppm(np.array([mass], dtype=np.float32))[0]
    else:
        mzs_val = mzs
        mass_val = mass

    # Find insertion index
    idx = np.searchsorted(mzs_val, mass_val)
    idx = np.clip(idx, 1, len(mzs_val) - 1)

    # Get nearest neighbors
    left = mzs_val[idx - 1]
    right = mzs_val[idx]

    # Return smallest difference
    return min(abs(left - mass_val), abs(right - mass_val))


def process_file(fl, imzML_dir, target_mass, unit):
    deviation = []
    p = ImzMLParser(os.path.join(imzML_dir, fl))

    for idx in range(len(p.coordinates)):
        mzs, _ = p.getspectrum(idx)
        smallest_diff = find_smallest_diff(mzs, target_mass, unit)
        deviation.append(smallest_diff)

    return deviation


def pre_last_dir(path):
    parts = path.rstrip('/').split(os.sep)
    return parts[-2] if len(parts) >= 2 else parts[-1]


# Helper for formatting median
def format_median(val, unit):
    if unit.lower() == 'da':
        return f"{val:.6f}"
    else:  # ppm
        return f"{val:.2f}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarizes the deviation to specific masses')
    parser.add_argument('mass_list', type=str, help="comma-separated list of masses")
    parser.add_argument('imzML_dir', type=str, default='', help='directory with imzML files')
    parser.add_argument('-comp_dir', type=str, default='', help='directory containing deviations to mass list as comparison')
    parser.add_argument('-result_dir', type=str, default='', help='directory to save output')
    parser.add_argument('-unit', type=str, default='ppm', help='unit')
    parser.add_argument('-plot', type=int, default=0, help='set to 1 for plotting')
    args = parser.parse_args()

    if args.plot == 0:
        matplotlib.use('Agg')

    try:
        mass_list = [float(x) for x in args.mass_list.split(',')]
    except ValueError:
        print("Error: All elements in mass list must be valid floats.")

    if args.result_dir == '':
        args.result_dir = os.path.join((args.imzML_dir), "quality_control")
    os.makedirs(args.result_dir, exist_ok=True)

    imzML_files = [f for f in os.listdir(args.imzML_dir) if os.path.isfile(os.path.join(args.imzML_dir, f))
                   and f.endswith('.imzML') and not f.startswith('.')]

    for mass in mass_list:
        deviation = []
        print("reading spectra and calculating deviation to mass {}".format(mass))

        results = Parallel(n_jobs=-1)(
            delayed(process_file)(fl, args.imzML_dir, mass, args.unit)
            for fl in tqdm(imzML_files)
        )

        deviation = np.asarray(
            [d for sublist in results for d in sublist],
            dtype=np.float32
        )

        # sns.boxplot(y=deviation, showfliers=False)
        # plt.title("Deviation to mass {} (median={:.6f})".format(mass, np.median(deviation)))
        # plt.savefig(os.path.join(args.result_dir, "deviation_to_{}.png".format(mass)))
        # if args.plot == 1:
        #     plt.show()
        # plt.close()

        # optional comparison of two deviations
        # --- Create figure with one or two boxplots ---
        plt.figure(figsize=(5, 6))

        # Data for boxplot
        data_to_plot = [deviation]

        labels = [pre_last_dir(args.result_dir)]
        labels[0] += f"\n(median: {format_median(np.median(deviation), args.unit)} {args.unit})"

        # Comparison deviation
        if args.comp_dir != '':
            comp_deviation_path = os.path.join(args.comp_dir, f"deviation_to_{mass}.npy")
            if os.path.exists(comp_deviation_path):
                comp_deviation = np.load(comp_deviation_path)
                data_to_plot.append(comp_deviation)
                labels.append(pre_last_dir(args.comp_dir))
                labels[-1] += f"\n(median: {format_median(np.median(comp_deviation), args.unit)} {args.unit})"
            else:
                comp_deviation = None

        # Compute median differences if two boxplots
        median_diff_text = ''
        if len(data_to_plot) == 2:
            median_dev_curr = np.median(deviation)
            median_dev_comp = np.median(comp_deviation)
            signed_diff = median_dev_curr - median_dev_comp
            abs_diff = abs(signed_diff)
            median_diff_text = f"\n Median diff {format_median(abs_diff, args.unit)} {args.unit}"
        else:
            median_diff_text = f"\nMedian {format_median(np.median(deviation), args.unit)} {args.unit}"

        # Plot boxplots
        plt.boxplot(data_to_plot, labels=labels, showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor='skyblue', color='black'),
                    medianprops=dict(color='red'))

        plt.ylabel(f"Deviation ({args.unit})")
        plt.title(f"Deviation to mass {mass}{median_diff_text}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save figure
        plt.savefig(os.path.join(args.result_dir, f"deviation_to_{mass}.png"), dpi=300)

        if args.plot == 1:
            plt.show()
        plt.close()

        np.save(os.path.join(args.result_dir, f"deviation_to_{mass}.npy"), deviation)

