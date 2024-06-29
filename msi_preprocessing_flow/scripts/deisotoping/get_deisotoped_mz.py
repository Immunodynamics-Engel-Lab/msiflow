from urllib.request import urlretrieve
import argparse
import numpy as np
#from pyopenms import *
import os
from pyimzml.ImzMLParser import ImzMLParser
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg.utils import get_combined_dataframe_from_files, booltoint


def find_nearest_val_in_list(val, lst):
    arr = np.asarray(lst)
    abs_val_arr = np.abs(arr - val)
    smallest_diff_ind = abs_val_arr.argmin()
    smallest_diff = abs_val_arr[smallest_diff_ind]
    return smallest_diff, smallest_diff_ind, arr[smallest_diff_ind]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs deisotoping on the mean spectrum of multiple MSI data')
    parser.add_argument('imzML_dir', type=str, help='directory with imzML files')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store deisotoped mz values')
    parser.add_argument('-tolerance', type=float, default=0.01, help='m/z tolerance in Da')
    parser.add_argument('-openMS', type=lambda x: booltoint(x), default=0, help='set to 1 to use openMS for deisotoping')
    parser.add_argument('-min_isotopes', type=int, default=2, help='minimum number of expected isotope peaks')
    parser.add_argument('-max_isotopes', type=int, default=6, help='maximum number of expected isotope peaks')
    parser.add_argument('-decreasing_model', type=bool, default=True, help='set to True to use decreasing model')
    parser.add_argument('-debug', type=bool, default=False, help='set to True for debugging')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join((args.imzML_dir), "deisotoped")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

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
                                            use_decreasing_model, start_intensity_check, add_up_intensity)
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
        mzs = mzs.tolist()
        ints = ints.tolist()
        mzs_deiso = []
        ints_deiso = []
        while mzs:
            mono_mz = mzs[0]
            mono_int = ints[0]
            pre_mz = mzs[0]
            pre_int = ints[0]
            # print('----------')
            # print(mzs)
            # print(mono_mz)
            for i in range(1, args.max_isotopes+1):
                if len(mzs) > 1:
                    smallest_diff, smallest_diff_ind, val = find_nearest_val_in_list(mono_mz + i, mzs)
                    diff_to_mono = smallest_diff - int(smallest_diff)
                    if diff_to_mono <= args.tolerance and ints[smallest_diff_ind] < pre_int:
                        # is an isotope --> remove from list and set as predecessor intensity
                        pre_int = ints[smallest_diff_ind]
                        mzs.pop(smallest_diff_ind)
                        ints.pop(smallest_diff_ind)
                        #print('removing {} from mzs'.format(val))
                    else:
                        # no isotope --> remove first element from list to carry on with next m/z and insert mono mz to mzs_deiso
                        mzs_deiso.append(mono_mz)
                        ints_deiso.append(mono_int)
                        mzs.pop(0)
                        ints.pop(0)
                        #print('removing {} from mzs'.format(mono_mz))
                        break
                elif len(mzs) == 1:
                    mzs_deiso.append(mono_mz)
                    ints_deiso.append(mono_int)
                    mzs.pop(0)
                    ints.pop(0)

        # print('--------result-------------')
        print('reduced m/z values from {} to {}'.format(len(mzs_save), len(mzs_deiso)))
        # print(ints)
        # print(mzs_deiso)
        # print(ints_deiso)

    if args.debug:
        plt.stem(mzs_copy, ints_copy, markerfmt=' ', basefmt=" ")
        #plt.plot(mzs, ints)
        plt.plot(mzs_deiso, ints_deiso, marker="o", ls="", ms=3)
        plt.show()

    # print(os.path.join(args.result_dir, 'deiso_mz.npy'))
    np.save(os.path.join(args.result_dir, 'deiso_mz.npy'), np.asarray(mzs_deiso))



#
# imzML = '/Users/philippaspangenberg/Desktop/test_brain/imzML/processed/processed/T019_Recall 716.imzML'
# cmz = True
#
# if cmz:
#     df = get_dataframe_from_imzML(imzML)
#     df_sum = get_summarized_spectrum(df, method='mean')
#     mzs = np.asarray(df_sum.columns.to_list())
#     int = df_sum.to_numpy().flatten()
#     print(mzs.shape)
#     thr = np.percentile(int, 25)
#     print(thr)
#     mzs = mzs[int > thr]
#     int = int[int > thr]
#     print(mzs.shape)
#     print(mzs)
#     print(int)
# else:
#     p = ImzMLParser(imzML)
#     mzs, int = get_mean_spectrum(p)
#
# spectrum = MSSpectrum()
# # mz = range(1500, 500, -100)
# # i = [0 for mass in mz]
# spectrum.set_peaks([mzs, int])
#
# # Sort the peaks according to ascending mass-to-charge ratio
# spectrum.sortByPosition()
#
# tolerance = 0.01
# ppm = False
# min_charge = 1
# max_charge = 1
# keep_only_deisotoped = True
# min_isotopes = 2
# max_isotopes = 8
# make_single_charged = False
# annotate_charge = True
# annotate_iso_peak_count = True
# use_decreasing_model = True
# start_intensity_check = 1
# add_up_intensity = False
#
# print(spectrum.size())
# Deisotoper.deisotopeAndSingleCharge(spectrum, tolerance, ppm, min_charge, max_charge, keep_only_deisotoped,
#                                     min_isotopes, max_isotopes,
#                                     make_single_charged, annotate_charge, annotate_iso_peak_count,
#                                     use_decreasing_model, start_intensity_check, add_up_intensity)
# print(spectrum.size())
#
#
# spectrum_deiso = MSSpectrum()
# # mz = range(1500, 500, -100)
# # i = [0 for mass in mz]
#
#
# mzs_deiso = []
# ints_deiso = []
# for mz, i in zip(*spectrum.get_peaks()):
#     mzs_deiso.append(mz)
#     ints_deiso.append(i)
# spectrum_deiso.set_peaks([mzs_deiso, ints_deiso])
#
#
# # markerline, stemlines, baseline = plt.stem(mzs, int, markerfmt=' ', basefmt=" ")
# # plt.setp(stemlines, 'linewidth', 0.5)
# plt.plot(mzs, int)
# plt.plot(mzs_deiso, ints_deiso, marker="o", ls="", ms=3)
# plt.show()
#
#
#
# # # Iterate over spectrum of those peaks
# # for p in spectrum:
# #     print(p.getMZ(), p.getIntensity())
#
# # # More efficient peak access with get_peaks()
# # for mz, i in zip(*spectrum.get_peaks()):
# #     print(mz, i)
# #
# # # Access a peak by index
# # print(spectrum[2].getMZ(), spectrum[2].getIntensity())



# gh = "https://raw.githubusercontent.com/OpenMS/pyopenms-docs/master"
# urlretrieve (gh + "/src/data/BSA1.mzML", "BSA1.mzML")
#
# min_charge = 1
# min_isotopes = 2
# max_isotopes = 10
# use_decreasing_model = True
# start_intensity_check = 3
#
#
# e = MSExperiment()
# MzMLFile().load("BSA1.mzML", e)
# s = e[214]
# s.setFloatDataArrays([])
# print(s)
# Deisotoper.deisotopeAndSingleCharge(s, 0.1, False, 1, 3, True,
#                                     min_isotopes, max_isotopes,
#                                     True, True, True,
#                                     use_decreasing_model, start_intensity_check, False)
#
# print(e[214].size())
# print(s.size())
#
# e2 = MSExperiment()
# e2.addSpectrum(e[214])
# MzMLFile().store("BSA1_scan214_full.mzML", e2)
# e2 = MSExperiment()
# e2.addSpectrum(s)
# MzMLFile().store("BSA1_scan214_deisotoped.mzML", e2)
#
# maxvalue = max([p.getIntensity() for p in s])
# for p in s:
#   if p.getIntensity() > 0.25 * maxvalue:
#     print(p.getMZ(), p.getIntensity())