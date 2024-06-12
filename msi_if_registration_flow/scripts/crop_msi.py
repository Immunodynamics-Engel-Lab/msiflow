import argparse
import os
import sys
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='crops msi data')
    parser.add_argument('imzML_fl', type=str, help='imzML file')
    #parser.add_argument('matrix_pixels', type=str, help='csv file with matrix pixel coordinates')
    parser.add_argument('x_min', type=int, help='x min/start')
    parser.add_argument('x_max', type=int, default=1, help='x max/stop')
    parser.add_argument('y_min', type=int, help='y min/start')
    parser.add_argument('y_max', type=int, default=1, help='y max/stop')
    parser.add_argument('-result_dir', type=str, default='', help='directory to store result')
    parser.add_argument('-plot', type=int, default=0, help='set to 1 to show plots')
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML_fl), "cropped")
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)


    p = ImzMLParser(args.imzML_fl)
    #pyx = (p.imzmldict["max count of pixels y"], p.imzmldict["max count of pixels x"])
    df = utils.get_dataframe_from_imzML(args.imzML_fl, multi_index=False)
    mzs = df.columns[2:].to_numpy()
    print(mzs.shape)
    print(df)

    df = df[(df['x'] >= args.x_min) & (df['x'] <= args.x_max) & (df['y'] >= args.y_min) & (df['y'] <= args.y_max)]
    df['x'] = df['x'] - args.x_min
    df['y'] = df['y'] - args.y_min
    print(df)
    print(df.columns[2:].to_numpy().shape)

    p = ImzMLParser(args.imzML_fl)
    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML_fl))) as writer:
        for i in tqdm(range(df.shape[0])):
            writer.addSpectrum(df.columns[2:].to_numpy(), df.iloc[i, 2:].to_numpy(),
                               (df.iloc[i, 0], df.iloc[i, 1], 0))

