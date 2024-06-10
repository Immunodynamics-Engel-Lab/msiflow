import tifffile
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage import restoration
import imgaug.augmenters as iaa
from tqdm import tqdm
import argparse
import sys

sys.path.append("..")
from pkg import utils


def preprocess(input_img, scale_ref_img='', radius=80, sigma=3, lower=0.0, upper=99.9, plot=False):
    # read data
    img_stack = tifffile.imread(input_img)

    if scale_ref_img != '':
        ref_img = tifffile.imread(scale_ref_img)
        imgaug_transformer = iaa.Sequential([], random_order=False)
        imgaug_transformer.append(iaa.Resize({"height": ref_img.shape[0], "width": ref_img.shape[1]}))
        img_stack_height = ref_img.shape[0]
        img_stack_width = ref_img.shape[1]
    else:
        img_stack_height = img_stack.shape[1]
        img_stack_width = img_stack.shape[2]

    proc_img_stack = np.empty((img_stack.shape[0], img_stack_height, img_stack_width)).astype('uint8')

    print('processing file ', input_img)
    for i in tqdm(range(img_stack.shape[0])):
        f, axarr = plt.subplots(ncols=4, sharex=True, sharey=True)

        # read image channel
        img = img_stack[i]

        # transform image shape
        if scale_ref_img != '':
            img = imgaug_transformer(image=img)

        # normalize data (range 0-1)
        img = utils.NormalizeData(img)

        counter = 0
        # plot raw image
        axarr[counter].imshow(img, cmap='gray')
        axarr[counter].set_title('raw')
        counter += 1

        # rolling ball background subtraction
        if radius != 0:
            background = restoration.rolling_ball(img, radius=radius)
            img = img - background
            # plot backsub image
            axarr[counter].imshow(img, cmap='gray')
            axarr[counter].set_title('background subtracted')
            counter += 1

        # gaussian blur
        if sigma != 0:
            img = gaussian(img, sigma=sigma)
            # plot gaussian image
            axarr[counter].imshow(img, cmap='gray')
            axarr[counter].set_title('gaussian blur')
            counter += 1

        # percentile normalization
        if lower != 0 or upper != 0:
            low, up = np.percentile(img, (lower, upper))
            img = rescale_intensity(img, in_range=(low, up))
            # plot intensity rescaled image
            axarr[counter].imshow(img, cmap='gray')
            axarr[counter].set_title('intensity rescale')
            counter += 1

        # plot image at individual preprocessing stages
        if plot:
            axarr[0].axis('off')
            axarr[1].axis('off')
            axarr[2].axis('off')
            axarr[3].axis('off')
            plt.show()

        # write processed image to stack
        proc_img_stack[i] = (img * 255).astype('uint8')

    return proc_img_stack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform image preprocessing')
    parser.add_argument('input', type=str, help='input image stack as TIF')
    parser.add_argument('output', type=str, default='', help='output image stack as TIF')
    parser.add_argument('-scale_ref_img', type=str, default='', help='image is downscaled to reference image shape')
    parser.add_argument('-radius', type=int, default=100, help='radius for rolling ball subtraction')
    parser.add_argument('-sigma', type=int, default=3, help='sigma of gaussian filter')
    parser.add_argument('-lower', type=float, default=0.0, help='lower limit of percentile stretching')
    parser.add_argument('-upper', type=float, default=99.9, help='upper limit of percentile stretching')
    parser.add_argument('-plot', type=bool, default=False, help='set to True for plotting')
    args = parser.parse_args()

    proc_img = preprocess(args.input, scale_ref_img=args.scale_ref_img, radius=args.radius, sigma=args.sigma,
                          lower=args.lower, upper=args.upper, plot=args.plot)
    tifffile.imwrite(args.output, proc_img, photometric='minisblack')



# img_stack = tifffile.imread(input)
# proc_img_stack = np.empty(img_stack.shape).astype('uint8')
#
# for i in range(img_stack.shape[0]):
#     print(i)
#     img = NormalizeData(img_stack[i])
#
#     # gaussian blur
#     img_gauss = gaussian(img, sigma=sigma)
#
#     # percentile normalization
#     low, up = np.percentile(img_gauss, (lower, upper))
#     img_resc = rescale_intensity(img_gauss, in_range=(low, up))
#     #img_resc = equalize_adapthist(img_gauss)
#     print(np.max(img_resc))
#     print(img_resc.max())
#
#     proc_img_stack[i] = (img_resc*255).astype('uint8')
#
#     if plot:
#         f, axarr = plt.subplots(ncols=3, sharex=True, sharey=True)
#         axarr[0].imshow(img, cmap='gray')
#         axarr[0].set_title('raw')
#         axarr[1].imshow(img_gauss, cmap='gray')
#         axarr[1].set_title('gaussian blur')
#         axarr[2].imshow(img_resc, cmap='gray')
#         axarr[2].set_title('intensity rescale')
#         axarr[0].axis('off')
#         axarr[1].axis('off')
#         axarr[2].axis('off')
#         plt.show()
#
# tifffile.imwrite(output, proc_img_stack)


