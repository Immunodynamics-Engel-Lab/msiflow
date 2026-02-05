import argparse
import os
import tifffile
# noinspection PyPackageRequirements
import ants
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# from pkg import utils

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10, 8)
Image.MAX_IMAGE_PIXELS = None  # To avoid decompression bomb images warning from Pillow


def get_overlay_img(image1, image2):
    image1 = (image1 - image1.min()) / (image1.max() - image1.min())
    image2 = (image2 - image2.min()) / (image2.max() - image2.min())

    height, width = image1.shape
    overlay_image = np.zeros((height, width, 3))
    overlay_image[..., 0] = image1  # Red channel
    overlay_image[..., 1] = image2  # Green channel
    overlay_image = (overlay_image * 255).astype(np.uint8)

    return overlay_image


def preview(img1, img2=None, title1='', title2='', figsize=(10, 8)):
    fig, axs = plt.subplots(2, figsize=figsize)
    _ = axs[0].imshow(np.array(img1), cmap='gray')
    # if img1_mask1 is not None:
    #     _ = axs[0].contour(img1_mask1)
    # if img1_mask2 is not None:
    #     _ = axs[0].contour(img1_mask2, colors='r')
    _ = axs[0].set_title(title1)

    if img2 is not None:
        _ = axs[1].imshow(np.array(img2), cmap='gray')
        # if img2_mask1 is not None:
        #     _ = axs[1].contour(img2_mask1)
        # if img2_mask2 is not None:
        #     _ = axs[1].contour(img2_mask2, colors='r')
        _ = axs[1].set_title(title2)
    plt.show()


def apply_transform(fixed_img, moving_img_stack, transformlist, interpolator="linear"):
    reg_transf_img_stack = np.zeros(moving_img_stack.shape)
    for i in range(moving_img_stack.shape[0]):
        image = moving_img_stack[i]
        reg_transf_img = ants.apply_transforms(fixed=ants.from_numpy(fixed_img),
                                               moving=ants.from_numpy(image.astype('float32')),
                                               transformlist=transformlist,
                                               interpolator=interpolator,  # gaussian > linear >'nearestNeighbor'
                                               )
        reg_transf_img_stack[i] = reg_transf_img.numpy()
    return reg_transf_img_stack


def registration(fixed_img, moving_img, transf_type, af_chan, out_dir, plot=False):
    fixed_fn = os.path.basename(fixed_img)
    moving_fn = os.path.basename(moving_img)
    af_chan -= 1

    fixed_img = tifffile.imread(fixed_img)
    moving_img_stack = tifffile.imread(moving_img)
    moving_img = moving_img_stack[af_chan]

    # fixed_img = utils.NormalizeData(fixed_img) * 255
    # moving_img = utils.NormalizeData(moving_img) * 255

    if moving_img.shape != fixed_img.shape:
        raise ValueError('fixed image and moving image do not have the same shape')

    if plot:
        preview(fixed_img, moving_img, 'fixed image', 'moving image')

    # # Registration
    # Using Symmetric Normalization transformation, optimising the mattes mutual information between the fixed and
    # moving images
    RegImage = ants.registration(ants.from_numpy(fixed_img),
                                 ants.from_numpy(moving_img),
                                 transf_type, syn_metric='mattes',)
    # Reg_Image is a dictionary containing the transformed image from moving to
    # fixed space (warpedmovout, and warpedfixout respectively) and vise versa, and the
    # coressponding transformations (fwdtransforms, and invtransforms respectively)
    # RegImage

    # # Transform the remaining images

    Fwd_trans = RegImage['fwdtransforms']

    # This function should receive both the fixed and moving images as ANTs image objects, or else it will return 1
    # Transforming the moving mask to the fixed space based on the transformation learned from registering the images
    reg_transf_img_stack = apply_transform(fixed_img=fixed_img, moving_img_stack=moving_img_stack,
                                           transformlist=Fwd_trans, interpolator="linear")
    tifffile.imwrite(os.path.join(out_dir, moving_fn), reg_transf_img_stack.astype('uint8'), photometric='minisblack')

    RegMovToFix = RegImage['warpedmovout'].numpy()
    #tifffile.imwrite(os.path.join(out_dir, 'reg_' + moving_fn), RegMovToFix.astype('uint8'), photometric='minisblack')

    # save overlay of fixed image (red image channel) and (registered) moving image (green image channel)
    overlay_before_reg = get_overlay_img(fixed_img, moving_img)
    overlay_after_reg = get_overlay_img(fixed_img, RegMovToFix)

    pil_overlay_before_reg = Image.fromarray(overlay_before_reg)
    pil_overlay_after_reg = Image.fromarray(overlay_after_reg)

    pil_overlay_before_reg.save(os.path.join(out_dir, moving_fn.split('.')[0] + '_before_registration_overlay.tif'),
                                format='TIFF')
    pil_overlay_after_reg.save(os.path.join(out_dir, moving_fn.split('.')[0] + '_after_registration_overlay.tif'),
                               format='TIFF')

    if plot:
        preview(fixed_img, RegMovToFix, title1='Fixed image', title2='Registered moving image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs registration '
                                                 '\nCaution: expects fixed and moving image to be same shape')
    parser.add_argument('fixed_img', type=str, help='path to fixed UMAP image')
    parser.add_argument('moving_img', type=str, help='path to moving image tif stack containing AF image')
    parser.add_argument('-transf_type', type=str, default='SyNRA', help='type of transformation')
    parser.add_argument('-af_chan', type=int, default=0, help='autofluorescence image channel')
    parser.add_argument('-out_file', type=str, default='', help='registered output file')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot')
    args = parser.parse_args()

    if args.out_file == '':
        out_dir = os.path.abspath(os.path.join(os.path.dirname(args.moving_img), 'registered'))
    else:
        out_dir = os.path.dirname(args.out_file)
    fn = os.path.basename(args.moving_img)
    args.out_file = os.path.join(out_dir, fn)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    registration(args.fixed_img, args.moving_img, args.transf_type, args.af_chan, out_dir, args.plot)


