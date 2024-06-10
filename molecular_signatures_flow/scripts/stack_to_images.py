import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf
import argparse
import sys

sys.path.append("..")
from pkg import utils


def write_images_from_stack(input_fl, output_dir, channels):
    fn = os.path.basename(input_fl)
    img_stack = tf.imread(input_fl)

    if not channels:
        channels = range(img_stack.shape[0])

    for i in range(img_stack.shape[0]):
        if i in channels:
            img = utils.NormalizeData(img_stack[i])
            tf.imwrite(os.path.join(output_dir, '{}_{}'.format(i+1, fn)), data=(img*255).astype('uint8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create individual TIF images from TIF image stack ')
    parser.add_argument('input', type=str, help='input TIF image stack')
    parser.add_argument('-out_dir', type=str, default='', help='output directory with individual images')
    parser.add_argument('-channels', type=lambda s: [int(item)-1 for item in s.split(',')], default=[],
                        help='pass delimited list of channels')
    args = parser.parse_args()

    if args.out_dir == '':
        args.out_dir = os.path.join(os.path.dirname(args.input), os.path.splitext(os.path.basename(args.input))[0])
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)


    write_images_from_stack(args.input, args.out_dir, args.channels)
    #tf.imwrite(args.output, img_stack, photometric='minisblack')


