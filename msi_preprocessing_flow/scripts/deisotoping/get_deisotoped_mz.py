from urllib.request import urlretrieve
import argparse
import pandas as pd
import numpy as np
from pyopenms import *
import os
from scipy.stats import pearsonr
from pyimzml.ImzMLParser import ImzMLParser
import matplotlib.pyplot as plt
import sys
from math import comb

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg.utils import get_combined_dataframe_from_files, booltoint


def overall_corr(comb_df, mono_mz, candidate_mz):
    """
    Compute the average Pearson correlation between two m/z columns in comb_df.
    Vectorized over all samples using NumPy.

    Args:
        comb_df: DataFrame with multi-index (sample x y) and m/z columns
        mono_mz: column name of the mono peak
        candidate_mz: column name of the candidate isotope peak

    Returns:
        float: average Pearson correlation
    """
    mono_vals = comb_df[mono_mz].to_numpy()
    cand_vals = comb_df[candidate_mz].to_numpy()

    # remove rows where both are zero to avoid NaN correlations
    mask = (mono_vals != 0) | (cand_vals != 0)
    mono_vals = mono_vals[mask]
    cand_vals = cand_vals[mask]

    if len(mono_vals) < 2:
        return 0.0  # not enough data to correlate

    # compute Pearson correlation manually (vectorized)
    mono_mean = mono_vals.mean()
    cand_mean = cand_vals.mean()
    numerator = ((mono_vals - mono_mean) * (cand_vals - cand_mean)).sum()
    denominator = np.sqrt(((mono_vals - mono_mean) ** 2).sum() * ((cand_vals - cand_mean) ** 2).sum())
    if denominator == 0:
        return 0.0
    return numerator / denominator


def find_nearest_val_in_list(val, lst):
    """
    Find nearest value in a list and return:
        - absolute difference
        - index
        - actual value
    """
    arr = np.asarray(lst)
    abs_val_arr = np.abs(arr - val)
    smallest_diff_ind = abs_val_arr.argmin()
    smallest_diff = abs_val_arr[smallest_diff_ind]
    return smallest_diff, smallest_diff_ind, arr[smallest_diff_ind]


def deisotope_peaks(mzs, ints, comb_df, args):
    """
    Deisotope a list of m/z and intensities using mass, correlation, and expected isotope intensity.

    Args:
        mzs: list of m/z values
        ints: list of intensities
        comb_df: dataframe with multi-index (sample x y) and m/z columns
        args: object with attributes:
            max_isotopes, tolerance, unit, corr_thr, score_thr
    Returns:
        mzs_deiso, ints_deiso
    """
    mzs_deiso = []
    ints_deiso = []

    # store metrics for every evaluated isotope candidate
    isotope_metrics = []

    while mzs:
        mono_mz = mzs[0]
        mono_int = ints[0]
        pre_int = mono_int
        accepted_any = False

        for i in range(1, args.max_isotopes + 1):

            if len(mzs) <= 1:
                break

            expected_mz = mono_mz + i * 1.0033548378    #13C isotope spacing

            # find peak closest to expected mz
            smallest_diff, smallest_diff_ind, candidate_mz = find_nearest_val_in_list(expected_mz, mzs)

            if args.unit == "ppm":
                mass_error = smallest_diff / expected_mz * 1e6
            else:
                mass_error = smallest_diff

            # Pearson correlation
            avg_corr = overall_corr(comb_df, mono_mz, candidate_mz)

            # intensity check against expected isotope intensity
            candidate_int = ints[smallest_diff_ind]
            intensity_ratio = candidate_int / pre_int
            intensity_ok = candidate_int <= pre_int and candidate_int >= args.intensity_thr * pre_int # example: between 25-100%

            accepted = (
                    mass_error <= args.tolerance and
                    avg_corr >= args.corr_thr and
                    intensity_ok
            )

            # save metrics
            isotope_metrics.append({
                "mono_mz": mono_mz,
                "mono_int": mono_int,
                "isotope_number": i,
                "expected_mz": expected_mz,
                "candidate_mz": candidate_mz,
                "candidate_int": candidate_int,
                "mass_error": mass_error,
                "correlation": avg_corr,
                "intensity_ratio": intensity_ratio,
                "intensity_ok": intensity_ok,
                "accepted": accepted,
            })

            if accepted:
                accepted_any = True
                pre_int = candidate_int
                mzs.pop(smallest_diff_ind)
                ints.pop(smallest_diff_ind)
            else:
                break

        # finalize monoisotopic peak
        mzs_deiso.append(mono_mz)
        ints_deiso.append(mono_int)
        mzs.pop(0)
        ints.pop(0)

    metrics_df = pd.DataFrame(isotope_metrics)

    return mzs_deiso, ints_deiso, metrics_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs deisotoping on the mean spectrum of multiple MSI data')
    parser.add_argument('imzML_dir', type=str, help='directory with imzML files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store deisotoped mz values')
    parser.add_argument('-tolerance', type=float, default=5, help='m/z tolerance in Da or ppm')
    parser.add_argument('-unit', type=str, default='ppm', help='unit')
    parser.add_argument('-corr_thr', type=float, default=0.5, help='pearson correlation threshold')
    parser.add_argument('-intensity_thr', type=float, default=0.25, help='min percentage intensity of pre-isotope')
    parser.add_argument('-openMS', type=lambda x: booltoint(x), default=0, help='set to 1 to use openMS for deisotoping')
    parser.add_argument('-min_isotopes', type=int, default=2, help='minimum number of expected isotope peaks')
    parser.add_argument('-max_isotopes', type=int, default=6, help='maximum number of expected isotope peaks')
    parser.add_argument('-decreasing_model', type=bool, default=True, help='set to True to use decreasing model')
    parser.add_argument('-debug', type=bool, default=False, help='set to True for debugging')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join((args.imzML_dir), "deisotoped")
    os.makedirs(args.result_dir, exist_ok=True)

    imzML_files = [f for f in os.listdir(args.imzML_dir) if os.path.isfile(os.path.join(args.imzML_dir, f))
                   and f.endswith('.imzML')]

    comb_df = get_combined_dataframe_from_files(args.imzML_dir, imzML_files, multi_index=True)
    # print(comb_df)

    mean_spec_df = comb_df.mean(axis=0).to_frame().T
    mzs = mean_spec_df.columns.to_numpy()
    ints = mean_spec_df.to_numpy().flatten()
    # print(mean_spec_df)
    # print(mzs)
    # print(ints)

    mzs_copy = mzs
    ints_copy = ints

    if args.openMS == 1:
        spectrum = MSSpectrum()
        spectrum.set_peaks([mzs, ints])

        # Sort the peaks according to ascending mass-to-charge ratio
        spectrum.sortByPosition()

        tolerance = args.tolerance
        if args.unit == 'ppm':
            ppm = True
        else:
            ppm = False
        min_charge = 1
        max_charge = 1
        keep_only_deisotoped = True
        min_isotopes = args.min_isotopes
        max_isotopes = args.max_isotopes
        make_single_charged = False
        annotate_charge = True
        annotate_iso_peak_count = True
        use_decreasing_model = args.decreasing_model
        start_intensity_check = 1
        add_up_intensity = False

        # print(spectrum.size())
        Deisotoper.deisotopeAndSingleCharge(spectrum, tolerance, ppm, min_charge, max_charge, keep_only_deisotoped,
                                            min_isotopes, max_isotopes,
                                            make_single_charged, annotate_charge, annotate_iso_peak_count,
                                            use_decreasing_model, start_intensity_check, add_up_intensity, None)
        # print(spectrum.size())

        spectrum_deiso = MSSpectrum()
        # mz = range(1500, 500, -100)
        # i = [0 for mass in mz]

        mzs_deiso = []
        ints_deiso = []
        for mz, i in zip(*spectrum.get_peaks()):
            mzs_deiso.append(mz)
            ints_deiso.append(i)
        spectrum_deiso.set_peaks([mzs_deiso, ints_deiso])

    else:
        # mzs = [720.59, 720.56, 721.60, 721.57, 722.61, 722.57, 723.60, 723.56, 724.56, 725.57]
        # ints = [10, 9, 12, 7, 6, 5, 4, 3, 2, 1]
        mzs_save = mzs.tolist()
        mzs_deiso, ints_deiso, metrics_df = deisotope_peaks(mzs.tolist(), ints.tolist(), comb_df, args)

        # print('--------result-------------')
        print('reduced m/z values from {} to {}'.format(len(mzs_save), len(mzs_deiso)))
        # print(ints)
        # print(mzs_deiso)
        # print(ints_deiso)
        metrics_df.to_csv(os.path.join(args.result_dir, "isotope_candidate_metrics.csv"), index=False)

    if args.debug:
        plt.stem(mzs_copy, ints_copy, markerfmt=' ', basefmt=" ")
        #plt.plot(mzs, ints)
        plt.plot(mzs_deiso, ints_deiso, marker="o", ls="", ms=3)
        plt.show()

    # print(os.path.join(args.result_dir, 'deiso_mz.npy'))
    np.save(os.path.join(args.result_dir, 'deiso_mz.npy'), np.asarray(mzs_deiso))
