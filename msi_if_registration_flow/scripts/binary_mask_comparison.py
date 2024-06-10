from sklearn.metrics import jaccard_score
from scipy import ndimage
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate overlay of 2 binary images')
    parser.add_argument('gt_file', type=str, help='path to ground truth image')
    parser.add_argument('mask_file', type=str, help='path of tif image stack containing binary mask')
    parser.add_argument('output_file', type=str, help='svg output file with mask overlay and Jaccard score')
    parser.add_argument('-mask_chan', type=int, default=0, help='channel of binary mask')
    parser.add_argument('-fill_holes', type=bool, default=False, help='set to True to fill holes in gt image')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot')
    args = parser.parse_args()

    gt = tifffile.imread(args.gt_file).astype('uint8')

    if args.fill_holes:
        gt = ndimage.binary_fill_holes(gt, structure=np.ones((5, 5))).astype(int)

    mask = tifffile.imread(args.mask_file)[args.mask_chan-1].astype('uint8')

    # normalize
    gt = gt - gt.min()
    gt = gt / gt.max()
    mask = mask - mask.min()
    mask = mask / mask.max()

    # threshold mask if not completely binary
    if np.any(mask) != 0 or np.any(mask) != 1:
        mask = mask > threshold_otsu(mask)

    # calculate Jaccard
    jacc = jaccard_score(y_true=gt.ravel(), y_pred=mask.ravel())

    # plot overlay with Jaccard score
    overlay = np.zeros(mask.shape, dtype=int)
    overlay[np.where((mask == 0) & (gt == 0))] = 0  # 0 = true negative
    overlay[np.where((mask == 1) & (gt == 1))] = 1  # 1 = true positive
    overlay[np.where((mask == 1) & (gt != 1))] = 2  # 2 = false positive
    overlay[np.where((mask != 1) & (gt == 1))] = 3  # 3 = false negative

    # unique, counts = np.unique(overlay, return_counts=True)
    # print(dict(zip(unique, counts)))

    color_list = {0: np.array([0, 0, 0]),  # true negative: black
                  1: np.array([255, 255, 0]),  # true positive: yellow
                  2: np.array([255, 0, 0]),  # false positive: red
                  3: np.array([0, 255, 0])}  # false negative: green

    overlay_rgb = np.ndarray(shape=(overlay.shape[0], overlay.shape[1], 3), dtype=int)
    for layer in range(3):
        for code in color_list.keys():
            overlay_rgb[:, :, layer][overlay == code] = color_list[code][layer]

    plt.imshow(overlay_rgb)
    plt.title('Jaccard: {:.4f}'.format(jacc))
    plt.axis('off')

    plt.savefig(args.output_file)

    if args.plot:
        plt.show()



