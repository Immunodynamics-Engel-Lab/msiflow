from pyimzml.ImzMLParser import ImzMLParser, getionimage
import tifffile as tif
import warnings
import os
import sys
import argparse

warnings.filterwarnings('ignore', module='pyimzml')

sys.path.append("..")
from pkg import utils
from pkg.plot import get_spec_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates binary image of measured tissue region from an imzML file')
    parser.add_argument('input_file', type=str, help='imzML input file')
    parser.add_argument('-out_dir', type=str, default='', help='output directory to save binary image '
                                                               'with measured pixels')
    args = parser.parse_args()

    fn = os.path.splitext(os.path.basename(args.input_file))[0]

    if args.out_dir == '':
        args.out_dir = os.path.join(os.path.dirname(args.input_file), 'meas_region')
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    spectra, coords = utils.get_dataframe_from_imzML(args.input_file, multi_index=True, get_coords=True)
    spectra.fillna(0, inplace=True)
    p = ImzMLParser(args.input_file)
    pyx = (p.imzmldict["max count of pixels y"] + 1, p.imzmldict["max count of pixels x"] + 1)

    spec_img = get_spec_img(pyx, coords)
    tif.imwrite(os.path.join(args.out_dir, fn + '.tif'), spec_img*255)
