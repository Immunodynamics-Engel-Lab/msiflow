from pyimzml.ImzMLParser import ImzMLParser, getionimage
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
import tifffile as tif
import warnings
import os
import sys
import argparse
import matplotlib.colors as mpc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg.plot import plot_ion_image

warnings.filterwarnings('ignore', module='pyimzml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots an ion image')
    parser.add_argument('mz', type=float, help='m/z value')
    parser.add_argument('input_file', type=str, help='imzML input file')
    parser.add_argument('-output_file', default='', type=str, help='ion image output file')
    parser.add_argument('-tol', default=0.1, type=float, help='tolerance')
    parser.add_argument('-unit', default='da', type=str, help='unit for tolerance, either da or ppm')
    parser.add_argument('-CLAHE', default=False, type=bool, help='set to True for CLAHE')
    parser.add_argument('-contrast_stretch', default=False, type=bool, help='set to True for contrast stretch')
    parser.add_argument('-lower', default=0.0, type=float, help='lower percentile for contrast stretching')
    parser.add_argument('-upper', default=99.9, type=float, help='upper percentile for contrast stretching')
    parser.add_argument('-plot', default=True, type=bool, help='set to True for plotting ion image')
    parser.add_argument('-cmap', default='viridis', type=str, help='colormap')
    parser.add_argument('-pyimzml', default=False, type=bool, help='if True, use pyimzml library')
    args = parser.parse_args()

    #args.cmap = mpc.LinearSegmentedColormap.from_list("", [(0, 0, 0, 1), (0.1177, 0.6275, 1, 1)])

    plot_ion_image(args.mz, args.input_file, args.output_file, args.tol, args.unit, args.CLAHE, args.contrast_stretch,
                   args.lower, args.upper, args.plot, args.cmap, args.pyimzml)
