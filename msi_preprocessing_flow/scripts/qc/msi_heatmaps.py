from pyimzml.ImzMLParser import ImzMLParser
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg.utils import to_mz, to_ppm


# plotting function
def plot_heatmap(data, title, label, out_dir, file_name, vmin=None, vmax=None, cmap='turbo', plot=False):
    plt.figure(figsize=(10, 8))
    if vmin and vmax:
        norm = None
    else:
        norm = colors.Normalize(vmin=data.min(), vmax=data.max())
    plt.imshow(
        data,
        cmap=cmap,
        interpolation='none',
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=label)
    plt.axis('off')
    plt.title(title)
    plt.savefig(os.path.join(out_dir, f'{file_name}_{label}.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'{file_name}_{label}.svg'), transparent=True)
    if plot:
        plt.show()


def get_limits(arr, percentile_clip=None):
    if percentile_clip is None:
        return None, None

    values = arr[arr > 0]  # ignore background
    if values.size == 0:
        return None, None

    return np.percentile(values, percentile_clip)


def plot_msi_qc_heatmaps(parser, out_dir, file_name, cmap='turbo', percentile_clip=(0.5, 99.5), plot=False):
    """
    Generate QC heatmaps for MSI data:
      1. Total Ion Current (TIC)
      2. Number of peaks per pixel
      3. Maximum intensity per pixel

    Parameters
    ----------
    parser : ImzMLParser
        Opened ImzMLParser object.
    cmap : str, default 'plasma'
        Colormap to use for heatmaps.
    use_log : bool, default False
        If True, use logarithmic color scaling to enhance visibility of small differences.
    """

    coords = np.array(parser.coordinates)
    x_max = coords[:, 0].max() + 1
    y_max = coords[:, 1].max() + 1

    # initialize heatmaps
    tic_map = np.zeros((y_max, x_max), dtype=float)
    peak_count_map = np.zeros((y_max, x_max), dtype=int)
    max_intensity_map = np.zeros((y_max, x_max), dtype=float)

    # iterate over pixels
    for idx, (x, y, z) in enumerate(tqdm(parser.coordinates, desc="Processing pixels")):
        mzs, intensities = parser.getspectrum(idx)
        tic_map[y, x] = intensities.sum()
        peak_count_map[y, x] = np.sum(intensities > 0)
        max_intensity_map[y, x] = intensities.max() if len(intensities) > 0 else 0

    # get min max values after percentile clipping
    tic_vmin, tic_vmax = get_limits(tic_map, percentile_clip)
    peaks_vmin, peaks_vmax = get_limits(peak_count_map, percentile_clip)
    max_vmin, max_vmax = get_limits(max_intensity_map, percentile_clip)

    # plot TIC heatmap
    plot_heatmap(tic_map, title='Total Ion Current (TIC) per pixel', label='tic', out_dir=out_dir, file_name=file_name, vmin=tic_vmin, vmax=tic_vmax, cmap=cmap, plot=plot)

    # plot number-of-peaks heatmap
    plot_heatmap(peak_count_map, title='Number of peaks per pixel', label='peaks', out_dir=out_dir, file_name=file_name, vmin=peaks_vmin, vmax=peaks_vmax, cmap=cmap, plot=plot)

    # plot maximum intensity heatmap
    plot_heatmap(max_intensity_map, title='Maximum intensity per pixel', label='max_intensity', out_dir=out_dir, file_name=file_name, vmin=max_vmin, vmax=max_vmax, cmap=cmap, plot=plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots QC heatmaps of an imzML file')
    parser.add_argument('imzML_fl', type=str, help='imzMl file')
    parser.add_argument('-result_dir', type=str, default='',
                        help='directory to store result, default \'\' to save results to directory called quality_control')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "quality_control")
    os.makedirs(args.result_dir, exist_ok=True)

    p = ImzMLParser(args.imzML_fl)
    fn = os.path.basename(args.imzML_fl).split('.')[0]
    plot_msi_qc_heatmaps(p, out_dir=args.result_dir, file_name=fn, percentile_clip=(0.1, 99.9), cmap='turbo', plot=args.plot)