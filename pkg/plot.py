import os.path
from matplotlib_venn import venn3_unweighted, venn2_unweighted, venn2, venn3
import numpy as np
import warnings
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from skimage.exposure import equalize_adapthist, rescale_intensity, equalize_hist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile
from skimage.morphology import remove_small_objects
import sys
from imzy import get_reader
from tqdm import trange
import numba
import seaborn as sns
from tqdm import tqdm

from pkg import utils


warnings.filterwarnings("ignore", category=UserWarning)


def plot_mz_umap(df, mz, out_file, df_control=None, show_neg_group_only=False, cmap='inferno', dot_size=1, plot=False):
    df[mz] = utils.NormalizeData(df[mz].to_numpy())

    colors = []
    cmap = plt.cm.get_cmap(cmap)
    # cmap = mpc.LinearSegmentedColormap.from_list("", ["#000000", "#1EA0FF"])
    # cmap = mpc.LinearSegmentedColormap.from_list("", [(0, 0, 0, 0), (1, 0, 0, 0)])
    for intensity in df[mz]:
        rgba = tuple(cmap(intensity))
        if intensity <= 0.99:
            rgba_list = list(rgba)
            rgba_list[3] = 1
            rgba = tuple(rgba_list)
        colors.append(rgba)
    df['color'] = colors

    if show_neg_group_only:
        fig = sns.scatterplot(x='UMAP_1', y='UMAP_2', data=df_control, color='gainsboro', s=dot_size, linewidth=0)

    for i in tqdm(np.arange(0, 1, 0.01)):
        perc = df[mz].quantile(i)
        df_perc = df[df[mz] > perc]
        sns.scatterplot(x=df_perc['UMAP_1'], y=df_perc['UMAP_2'], c=df_perc['color'].to_numpy(),
                        s=dot_size, linewidth=0)

    plt.axis('off')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # print('saving figure')
    plt.savefig(out_file, dpi=300, transparent=True)

    if plot:
        plt.show()


@numba.njit()
def find_between(data: np.ndarray, min_value: float, max_value: float):
    """Find indices between windows."""
    return np.where(np.logical_and(data >= min_value, data <= max_value))[0]


def get_ion_img_from_d(path, peak, window):
    reader = get_reader(path)
    peak_min, peak_max = peak - window, peak + window

    res = np.zeros(reader.get_n_pixels(), dtype=np.float32)
    for i, frame_id in enumerate(trange(1, reader.get_n_pixels(), desc="Extracting peak...", miniters=50)):
        indices, y = reader.read_centroid_spectrum(frame_id)
        x = reader.index_to_mz(frame_id, indices)
        res[i] = y[find_between(x, peak_min, peak_max)].sum()

    # now we can insert the array of intensities into an ion image
    array = np.zeros((reader.y_size, reader.x_size), dtype=np.float32)
    array[reader.y_coordinates_min, reader.x_coordinates_min] = res

    # plt.imshow(array)
    # plt.show()
    return array


def get_mz_img(pyx, msi_df, mz, tol=0.0):
    lower = mz - tol
    upper = mz + tol
    mzs_cols = msi_df.columns.to_numpy()
    mzs_cols_tol_range = np.where((mzs_cols >= lower) & (mzs_cols <= upper))
    # print(mzs_cols_tol_range)
    msi_df = msi_df.iloc[:, mzs_cols_tol_range[0]]
    msi_df['sum'] = msi_df.sum(axis=1)
    # print(msi_df)
    coords = msi_df.index.tolist()
    msi_img = np.zeros(pyx)
    for x_val, y_val in coords:
        msi_img[y_val, x_val] = msi_df.loc[(x_val, y_val), 'sum']
    return msi_img


def plot_img_heatmap(grayscale, output_file='', plot=False, cmap='jet', ticks=False):
    """
    Plots a grayscale image with colormap and colorbar

    :param grayscale: grayscale image as 2D numpy array
    :param output_file: file to save result
    :param plot: set to True if image should be plotted
    """
    max = np.max(grayscale)
    min = np.nanmin(np.array(grayscale)[grayscale != np.nanmin(grayscale)])

    # if cmap == 'jet':
    #     cmap = plt.cm.get_cmap('jet').copy()
    # else:
    #     cmap = plt.cm.get_cmap("Spectral").copy()
    cmap = plt.cm.get_cmap(cmap).copy()

    cs = plt.imshow(grayscale, interpolation='none', cmap=cmap)
    cs.cmap.set_under('w')
    cs.set_clim(min, max)
    cb = plt.colorbar(cs)
    if not ticks:
        cb.set_ticks([])
    plt.axis('off')

    if output_file != '':
        plt.savefig(output_file, transparent=True)
    if plot:
        plt.show()
    plt.close()


def construct_spot_image(imzML_file, vals, output_file='', cmap='Spectral'):
    """
    Constructs an image of a MSI data set (from an imzML file) with specific color values for each spot.

    :param imzML_file: imzML file to construct image from
    :param vals: values of spots
    :param output_file: spot image file path
    :type imzML_file: str
    :type vals: numpy array of shape (y, x, 1) will take values as labels and use spectral cmap as color code
    or numpy array of shape (y, x, 3) will take values as RGB color code
    :type output_file: str
    :return: spot image
    :rtype: numpy array of shape (y, x, 1) or (y, x, 3)
    :raises: UserWarning if wrong array dimension
    """
    p = ImzMLParser(imzML_file)

    # print("vals=",vals)
    # print("vals.shape=", vals.shape)
    # print("vals.ndim=", vals.ndim)

    if vals.ndim == 2:  # 3D RGB image
        im = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1, 3))
        for i, (x, y, z_) in enumerate(p.coordinates):
            im[y, x, 0] = vals[i, 0]
            im[y, x, 1] = vals[i, 1]
            if vals.shape[1] == 2:
                im[y, x, 2] = 0
            else:
                im[y, x, 2] = vals[i, 2]
        # plot RGB image
        plt.imshow(im, interpolation='none')
        plt.axis('off')
        if output_file != '':
            plt.savefig(output_file)
        plt.close()
    elif vals.ndim == 1:    # 2D grayscale image
        im = np.zeros((p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1))
        for i, (x, y, z_) in enumerate(p.coordinates):
            im[y, x] = vals[i]
        if output_file != '':
            if output_file.split('.')[1] == 'tif':
                im = utils.NormalizeData(im)
                tifffile.imwrite(output_file, (im*255).astype('uint8'))
            else:
                plot_img_heatmap(im, output_file, cmap=cmap)
    else:
        warnings.warn('cannot construct image from {} dimensional array'.format(vals.ndim), UserWarning)
    return im


def plot_ion_image(mz, input_file, output_file='', tol=0.1, unit='da', CLAHE=True, contrast_stretch=False, lower=0,
                   upper=99, plot=False, cmap='viridis', pyimzml=True, remove_isolated_px=False, output_dir=''):
    """
    Plots and saves an ion image from an imzML file.

    :param input_file: MSI file path (imzML or .d) to plot ion image from
    :param mz: m/z value
    :param output_file: ion image file path
    :param CLAHE: if set to True, do CLAHE
    :param plot: if set to True, plot ion image
    :type imzML_file: str
    :type mz: float
    :type output_file: str
    :type CLAHE: bool
    :type plot: bool
    """
    # print("input_file: ", input_file)
    # print("mz: ", mz)
    # print("output_file:", output_file)
    # print("output_dir:", output_dir)
    # print("tol:", tol)
    # print("unit:", unit)
    # print("CLAHE:", CLAHE)
    # print("contrast_stretch:", contrast_stretch)
    # print("lower: ", lower)
    # print("upper: ", upper)
    # print("plot: ", plot)
    # print("cmap: ", cmap)
    # print("pyimzml: ", pyimzml)
    # print("remove_isolated_px: ", remove_isolated_px)

    org_tol = tol

    if unit == 'ppm':
        da_diff = mz + (mz * (tol / 1000000 - 1))
        tol = da_diff

    if input_file.split('.')[1] == 'd':
        ion_img = get_ion_img_from_d(input_file, mz, tol)
    else:
        p = ImzMLParser(input_file)

        if not pyimzml:
            pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)
            msi_df = utils.get_dataframe_from_imzML(input_file, multi_index=True)
            ion_img = get_mz_img(pyx, msi_df, mz, tol)
        else:
            ion_img = getionimage(p, mz, tol=tol)

    # remove isolated pixels
    if remove_isolated_px:
        fg = ion_img > 0
        cleaned = remove_small_objects(fg, min_size=3) * 1
        ion_img[cleaned == 0] = 0

    if CLAHE:
        ion_img = equalize_adapthist(utils.NormalizeData(ion_img))   # CLAHE

    if contrast_stretch:
        # Contrast stretching
        low, up = np.percentile(ion_img, (lower, upper))
        ion_img = rescale_intensity(ion_img, in_range=(low, up))

    format = 'tif'
    if (output_file != '' or output_dir != '') and np.max(ion_img) != 0:
        if output_file == '':
            fn = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, fn + '_' + str(round(mz, 4)).replace('.', '_') + '.tif')
            #output_file = os.path.join(output_dir, fn + '_' + str(round(mz, 4)).replace('.', '_') + '.png')
        format = os.path.splitext(output_file)[-1][1:]
        # print("output_file=", output_file)
        # print("format=", format)
        if format == 'tif':
            tifffile.imwrite(output_file, (utils.NormalizeData(ion_img)*255).astype('uint8'))
        elif format == 'png':
            plt.imsave(output_file, arr=ion_img, cmap=cmap, dpi=300)
    if np.max(ion_img) == 0:
        print("WARNING: image {} has no positive pixels for {} m/z".format(os.path.splitext(
            os.path.basename(input_file))[0], round(mz, 4)))

    if plot:
        plt.imshow(ion_img, cmap=cmap)
        if unit == 'Da':
            plt.title(str(mz) + ' m/z ± ' + str(tol) + ' Da')
        else:
            plt.title(str(mz) + ' m/z ± ' + str(org_tol) + ' ppm')
        plt.colorbar()
        plt.axis('off')

        if output_file != '':
            if format != 'tif' and format != 'png':
                plt.savefig(output_file, dpi=300)
        plt.show()
    plt.close()


def plot_spectrum(mzs, intensities, plot=False, output_file=''):
    plt.plot(mzs, intensities)
    plt.xlabel('m/z [Da]')
    plt.ylabel('Intensities [a.u.]')
    if output_file != '':
        plt.savefig(output_file)
    if plot:
        plt.show()
    plt.close()


def scatterplot_3D(data, filepath, c, cmap='Spectral', size=10, plot=False):
    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=c, cmap=cmap, s=size)
    plt.savefig(filepath)
    if plot:
        plt.show()
    plt.close()


def scatterplot_2D(data, filepath, c, cmap='Spectral', size=10, plot=False):
    if c is not None:
        if np.min(c) == 0:
            clustered = c > 0
            plt.scatter(data[clustered, 0], data[clustered, 1], c=c[clustered], s=size, cmap=cmap)
        else:
            plt.scatter(x=data[:, 0], y=data[:, 1], c=c, cmap=cmap, s=size)
    else:
        plt.scatter(x=data[:, 0], y=data[:, 1], c=c, cmap=cmap, s=size)
    plt.savefig(filepath)
    if plot:
        plt.show()
    plt.close()


def contrast_stretching(image):
    upper_thr = np.percentile(image, 95)
    lower_thr = np.percentile(image, 5)
    image[image > upper_thr] = upper_thr
    image[image < lower_thr] = lower_thr
    return (image - lower_thr) / (upper_thr - lower_thr)


def plot_venn2(label1, label2, data1, data2, title, output_file='', plot=False, weighted=True):
    """
    Plots a Venn diagram of 2  sets

    :param label1: label of first set
    :param label2: label of second set
    :param data1: first data set
    :param data2: second data set
    :param title: title of figure
    :param output_file: output file to save plot
    :param plot: set to True to plot Venn diagram
    """
    label1 = label1 + '\n({})'.format(len(data1))
    label2 = label2 + '\n({})'.format(len(data2))
    inter_1_2 = data1 & data2
    data1 = data1.difference(inter_1_2)
    data2 = data2.difference(inter_1_2)

    if weighted:
        venn2(subsets=(len(data1), len(data2), len(inter_1_2)), set_labels=(label1, label2))
    else:
        venn2_unweighted(subsets=(len(data1), len(data2), len(inter_1_2)), set_labels=(label1, label2))

    plt.title(title)
    if output_file != '':
        plt.savefig(output_file)
    if plot:
        plt.show()
    plt.close()


def get_spec_img(pyxf, coords_f):
    """Get the binary image representing the selected spectra according to their coordinates

	Args:
		pyxf: shape of the spectral image
		coords_f: coordinates of selected spectra within the image
	"""

    spec_img_f = np.zeros(pyxf).astype(np.uint8)
    for xf, yf in coords_f:
        spec_img_f[yf, xf] = 1
    return spec_img_f



