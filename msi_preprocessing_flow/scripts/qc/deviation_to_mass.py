import argparse
import os
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def find_smallest_diff(mzs, mass):
    idx = np.searchsorted(mzs, mass)
    idx = np.clip(idx, 1, len(mzs) - 1)
    left = mzs[idx - 1]
    right = mzs[idx]
    return min(abs(left - mass), abs(right - mass))


def process_file(fl, imzML_dir, target_mass):
    deviation = []
    p = ImzMLParser(os.path.join(imzML_dir, fl))

    for idx in range(len(p.coordinates)):
        mzs, _ = p.getspectrum(idx)
        smallest_diff = find_smallest_diff(mzs, target_mass)
        deviation.append(smallest_diff)

    return deviation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarizes the deviation to specific masses')
    parser.add_argument('mass_list', type=str, help="comma-separated list of masses")
    parser.add_argument('imzML_dir', type=str, default='', help='directory with imzML files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to save output')
    parser.add_argument('-plot', type=int, default=0, help='set to 1 for plotting')
    args = parser.parse_args()

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
            delayed(process_file)(fl, args.imzML_dir, mass)
            for fl in tqdm(imzML_files)
        )

        deviation = np.asarray(
            [d for sublist in results for d in sublist],
            dtype=np.float32
        )

        sns.boxplot(y=deviation, showfliers=False)
        plt.title("Deviation to mass {} (median={:.6f})".format(mass, np.median(deviation)))
        plt.savefig(os.path.join(args.result_dir, "deviation_to_{}.png".format(mass)))
        if args.plot == 1:
            plt.show()
        plt.close()

