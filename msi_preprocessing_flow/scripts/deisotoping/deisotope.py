from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np
import argparse
import os
from pyimzml.ImzMLWriter import ImzMLWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs deisotoping to a mz list')
    parser.add_argument('imzML_fl', type=str, help='imzMl file')
    parser.add_argument('deiso_mz_fl', type=str, help='file with deisotoped mz as numpy array')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "deisotoped")
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)

    # load deiso mz
    deiso_mz = np.load(args.deiso_mz_fl).astype(np.float32)
    print("deisotoped m/z spectrum has {} bins".format(deiso_mz.shape))

    # reduce data to deisotoped mz
    p = ImzMLParser(args.imzML_fl)
    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML_fl))) as writer:
        for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
            mzs, intensities = p.getspectrum(idx)
            mzs = mzs.astype(np.float32)
            deiso_idx = np.where(np.in1d(mzs, deiso_mz))[0]
            writer.addSpectrum(mzs[deiso_idx], intensities[deiso_idx], (x, y, z))
