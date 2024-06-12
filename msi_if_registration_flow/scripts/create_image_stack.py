import os
import numpy as np
import tifffile as tf
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils


def write_img_stack_from_img_dir(img_dir, output_fl):
    fn = sorted([f for f in os.listdir(img_dir) if (f.endswith('tif') or f.endswith('.tiff')) and not f.startswith('.')])
    #cn = pd.read_csv(channels_file, sep='\t', header=None).values[:, 0]
    cn = [c.split('.')[0].split('_')[0] for c in fn]    # expects file names: channel_*.tif

    img_stack = []
    for chan, fl in zip(cn, fn):
        img = utils.NormalizeData(tf.imread(os.path.join(img_dir, fl)))
        img_stack.append((img*255).astype('uint8'))
        #img_stack.append(tifffile.imread(os.path.join(img_dir, chan + '.tif')))
    img_stack = np.asarray(img_stack)

    tf.imwrite(output_fl, img_stack, photometric='minisblack')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create image stack TIF files from individual TIF images')
    parser.add_argument('input', type=str, help='input directory with TIF images of sample')
    parser.add_argument('output', type=str, default='', help='output image stack as TIF')
    args = parser.parse_args()

    write_img_stack_from_img_dir(args.input, args.output)


