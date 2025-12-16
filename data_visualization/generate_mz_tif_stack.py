import numpy as np
import tifffile as tiff
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg.plot import plot_ion_image, get_mz_img
from pkg import utils


def imzml_to_ome_tiff(imzml_path, output_ome_tiff):
    p = ImzMLParser(imzml_path)
    msi_df = utils.get_dataframe_from_imzML(imzml_path, multi_index=True)
    print(msi_df.head())

    pyx = (p.imzmldict["max count of pixels y"] + 1, p.imzmldict["max count of pixels x"] + 1)

    stack = np.zeros((len(msi_df.columns), pyx[0], pyx[1]), dtype=np.float32)

    print("Extracting ion images...")
    for i, mz in enumerate(tqdm(msi_df.columns)):
        img = get_mz_img(pyx, msi_df, mz, tol=0.0)
        stack[i] = img.astype(np.uint8)

    # Reshape to OME: (T, Z, C, Y, X)
    ome_data = stack[np.newaxis, np.newaxis, :, :, :]

    # Channel names = m/z values
    channel_names = [f"m/z {mz:.5f}" for mz in msi_df.columns]

    tiff.imwrite(
        output_ome_tiff,
        ome_data,
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "TZCYX",
            "Channel": {"Name": channel_names},
            "PhysicalSizeX": 1.0,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        }
    )

    print(f"Saved OME-TIFF to: {output_ome_tiff}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tif stack with m/z values of an imzML file')
    parser.add_argument('input_file', type=str, help='imzML input file')
    parser.add_argument('output_file', default='', type=str, help='ome.tif output file')
    args = parser.parse_args()

    imzml_to_ome_tiff(imzml_path=args.input_file, output_ome_tiff=args.output_file)

