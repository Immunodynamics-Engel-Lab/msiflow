import tifffile
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

def norm(img):
    img = img - img.min()
    img = img / img.max()
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combines 2 binary images')
    parser.add_argument('file1', type=str, help='first file path (can be stack of images)')
    parser.add_argument('file2', type=str, help='second file path')
    parser.add_argument('output', type=str, help='output file path')
    parser.add_argument('-chan_list', type=lambda s: [int(item) - 1 for item in s.split(',')], default=[],
                        help='if file1 is an image stack, pass delimited list of image channels '
                             'where logical operator should be applied')
    parser.add_argument('-logical_operator', type=str, default='and',
                        choices=['and', 'and_not', 'or'], help='second file path')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot')
    args = parser.parse_args()

    fn = os.path.splitext(os.path.basename(args.file1))[0]

    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # read data
    img1 = tifffile.imread(args.file1)
    img2 = tifffile.imread(args.file2)

    # if not provided apply logical operation on all images of image stack
    if img1.squeeze().ndim > 2 and not args.chan_list:
        args.chan_list = range(img1.shape[0])

    res_img = np.zeros(img1.shape)

    # logical and
    if args.logical_operator == 'and' or args.logical_operator == 'and_not':
        if args.logical_operator == 'and_not':
            img2 = np.invert(img2)
        # process stack
        if img1.squeeze().ndim > 2:
            for i in range(img1.shape[0]):
                if i in args.chan_list:
                    res_img[i] = ((img1[i] > 0) & (img2 > 0))*255
                    if args.plot:
                        f, axarr = plt.subplots(ncols=3, sharex=True, sharey=True)
                        axarr[0].imshow(img1[i], cmap='gray')
                        axarr[0].set_title('img1')
                        axarr[1].imshow(img2, cmap='gray')
                        axarr[1].set_title('img2')
                        axarr[2].imshow(res_img[i], cmap='gray')
                        axarr[2].set_title('img1 {} img2'.format(args.logical_operator))
                        axarr[0].axis('off')
                        axarr[1].axis('off')
                        axarr[2].axis('off')
                        plt.show()
                else:
                    res_img[i] = img1[i]
        # process single image
        else:
            res_img = ((img1 > 0) & (img2 > 0))*255
            if args.plot:
                f, axarr = plt.subplots(ncols=3, sharex=True, sharey=True)
                axarr[0].imshow(img1, cmap='gray')
                axarr[0].set_title('img1')
                axarr[1].imshow(img2, cmap='gray')
                axarr[1].set_title('img2')
                axarr[2].imshow(res_img, cmap='gray')
                axarr[2].set_title('img1 {} img2'.format(args.logical_operator))
                axarr[0].axis('off')
                axarr[1].axis('off')
                axarr[2].axis('off')
                plt.show()

    else:
        res_img = img1

    tifffile.imwrite(args.output, res_img.astype('uint8'), photometric='minisblack')



    # #result_img = np.zeros(img1.shape)
    # result_img = img1
    #
    # #result_img[img1 == 1] = 1
    # result_img[img2 == 1] = 0
    # #result_img[(img1 == 1) & (img2 == 0)] = 1
    #
    # tifffile.imwrite(args.out_file, (result_img*255).astype('uint8'))
    # #tifffile.imwrite(out_file, (tissue_img*255).astype('uint8'))
