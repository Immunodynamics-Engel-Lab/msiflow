import numpy as np
import argparse
import os
import sys
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg import utils


def get_scfactor(spectra, method='median', reference=None):
    if method == 'mfc':
        ratios = np.divide(spectra, reference)      # ratios between intensities and reference intenisties
        scfactor = np.nanmedian(ratios.flatten())   # median of ratios
    elif method == 'mean':
        scfactor = np.nanmean(spectra.flatten())
    elif method == 'sum':
        scfactor = np.nansum(spectra.flatten())
    else:
        scfactor = np.nanmedian(spectra.flatten())
    return scfactor


if __name__ == '__main__':
    # spec = np.array([[1, 5, 2],
    #                  [2, 6, 3],
    #                  [2, 5, 1],
    #                  [3, 8, 2],
    #                  [3, 7, 3]])
    # ref = np.array([2, 6, 2])
    # sc_fac = get_scfactor(spec, method='mfc', reference=ref)
    # print(sc_fac)
    parser = argparse.ArgumentParser(description='Performs internormalization of MSI data')
    parser.add_argument('imzML_fl', type=str, help='imzML file')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result')
    parser.add_argument('-method', type=str, default='median', help='method for normalization')
    parser.add_argument('-reference', type=str, default='', help='reference for mfc normalisation')
    parser.add_argument('-debug', type=bool, default=False, help='set to True for debugging')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "internormed")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    p = ImzMLParser(args.imzML_fl)
    df = utils.get_dataframe_from_imzML(args.imzML_fl, multi_index=True)
    # print(df)

    # ignore mz features and spectra with all zeros or nans
    df = df.replace(0, np.nan)
    df = df.dropna(how='all', axis='columns')
    # df = df.dropna(how='all', axis=0)
    # df = df.dropna(how='all', axis=1)

    # get scaling factors for spectr
    spec = df.to_numpy()
    if args.reference != '':
        reference = np.load(args.reference)
    else:
        reference = None
    scfac = get_scfactor(spec, args.method, reference)

    # set divisor to 1 if scaling factor is 0 to prevent divide by 0 error
    if np.isnan(scfac) or scfac == 0:
        print("WARNING: internorm scaling factor is set to 1")
        scfac = 1

    # apply scaling factor
    df = df.replace(np.nan, 0)
    df_norm = df.divide(scfac)
    # print(scfac)

    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML_fl))) as writer:
        for index, row in df_norm.iterrows():
            writer.addSpectrum(df_norm.columns.to_numpy(), row.to_numpy(), (index[0], index[1], 0))

