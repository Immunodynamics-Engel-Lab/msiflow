import argparse
import os
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Summarizes the deviation to specific mass')
    parser.add_argument('mass', type=float, help='mass list')
    parser.add_argument('imzML_dir', type=str, default='', help='directory with imzML files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to save output')
    parser.add_argument('-plot', type=int, default=0, help='set to 1 for plotting')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join((args.imzML_dir), "quality_control")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    imzML_files = [f for f in os.listdir(args.imzML_dir) if os.path.isfile(os.path.join(args.imzML_dir, f))
                   and f.endswith('.imzML') and not f.startswith('.')]
    imzML_paths = [os.path.join(args.imzML_dir, f) for f in imzML_files]

    deviation = []
    print("reading spectra and calculating deviation to mass {}".format(args.mass))
    for fl in tqdm(imzML_files):
        p = ImzMLParser(os.path.join(args.imzML_dir, fl))
        for id in range(len(p.coordinates)):
            mzs, _ = p.getspectrum(id)
            smallest_diff, _, _ = utils.find_nearest_value(args.mass, mzs)
            deviation.append(smallest_diff)
    deviation = np.asarray(deviation).astype(np.float32)

    sns.boxplot(y=deviation, showfliers=False)
    plt.title("Deviation to mass {} (median={:.6f})".format(args.mass, np.median(deviation)))
    plt.savefig(os.path.join(args.result_dir, "deviation_to_{}.png".format(args.mass)))
    if args.plot == 1:
        plt.show()

