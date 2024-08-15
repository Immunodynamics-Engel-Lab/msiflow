#!/bin/sh

# run msi_if_registration_flow to register IFM and MSI
snakemake --snakefile msi_if_registration_flow/Snakefile --configfile demo/data/msi_if_registration/config.yaml --config data='demo/data/msi_if_registration' --cores all

# copy the registered IFM image to demo/data/if_segmentation
cp demo/data/msi_if_registration/registered/UPEC_12.tif demo/data/if_segmentation

# run if_segmentation_flow to segment the Ly6G image channel
snakemake --snakefile if_segmentation_flow/Snakefile --configfile demo/data/if_segmentation/config.yaml --config data='demo/data/if_segmentation' --cores all

# copy the registered and segmented image to demo/data/Ly6G_signatures/bin_imgs
mkdir demo/data/Ly6G_signatures/bin_imgs
cp demo/data/if_segmentation/segmented/UPEC_12.tif demo/data/Ly6G_signatures/bin_imgs

# run molecular_signatures_flow to extract lipidomic signatures of Ly6G (neutrophils)
snakemake --snakefile molecular_signatures_flow/Snakefile --configfile demo/data/Ly6G_signatures/config.yaml --config data='demo/data/Ly6G_signatures' --cores all