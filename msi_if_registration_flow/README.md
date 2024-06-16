# Workflow for MSI & IFM image co-registration
This snakemake workflow combines MSI and IFM data via image co-registration. It creates a UMAP image from MSI as fixed
image and the autofluorescence image from IFM as moving image. See workflow DAG 
[here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/msi_if_registration_flow/dag.pdf).

## Installation
please see installation instructions [here](https://github.com/Immunodynamics-Engel-Lab/msiflow).

## Run
To run this workflow via **Docker** follow these instructions:
  - start Docker
  - in a terminal run `docker run -v <path-to-data-and-config>:/home/user/msiflow/data -e 'WORKFLOW=msi_if_registration_flow' -e 'CORES=<number-of-cores>' phispa1812/msiflow`

To run this workflow **locally** follow these instructions:
- in a terminal navigate to the root directory of msiFlow
- if you have installed the packages in a conda environment, activate the environment via `conda activate msiflow_env`
- run `snakemake --snakefile msi_if_registration_flow/Snakefile --cores <number-of-cores> --configfile <path-to-config>`

In the commands above
- enter the path of your data folder for `<path-to-data-and-config>`. See below how to structure and 
name your files in your data folder for successful completion of msiFlow.
- enter the max. number of cores to be used by msiFlow for `<number-of-cores>`. To provide all cores type *all*.
- enter the path of your configuration file for `<path-to-config>` when using the **local** version. 

## Input data
#### Data folder
Your data folder must contain the following directories and files:
- directory named *fixed* containing imzML files 
- directory named *moving* containing a subdirectory for each IFM sample containing TIF images
- optional subdirectory *fixed/mask* containing binary masks from MSI to validate the registration result 
  (in that case IFM must also contain a binary image channel defined in the config)
- *config.yaml* when using the **Docker** version of msiFlow

#### Configuration file
You must provide a **configuration file** named *config.yaml* to run this workflow. All parameters used by the workflow are defined in
this configuration file. See the [wiki]() for a description of all parameters. An example configuration file can be
found [here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/msi_segmentation_flow/data/config.yaml).

When using the **Docker** version of msiFlow the *config.yaml* must be provided in the data folder. 

For the **local** version of msiFlow the path of the configuration file can be specified in the command.

**Example:**
```
data
|   config.yaml
└─ fixed
|   |   group_sampleno.ibd
|   |   group_sampleno.imzML
|   |   ...
|   └─ mask
|   |   group_sampleno.tif
|   |   ...
└─ moving
    └─ group_sampleno
    └─ ...
```

## Output
The workflow outputs the following directories and files:
- subdirectory of the *fixed* directory named *umap* containing UMAP images from MSI
- subdirectory of the *moving* directory named *preprocessed* containing the pre-processed IFM image stacks 
- directory named *registered* containing the registered IFM images and overlay of masks if provided

**Example:**
```
data
|   config.yaml
└─ fixed
|   |   group_sampleno.ibd
|   |   group_sampleno.imzML
|   |   ...
|   └─ mask
|   |   group_sampleno.tif
|   |   ...
|   └─ umap
|       |   umap_grayscale_group_sampleno.tif
|       |   umap_heatmap_group_sampleno.svg
|       |   ...
└─ moving
|   └─ group_sampleno
|   └─ ...
|   |   group_sampleno.tif
|   |   ...
|   └─ preprocessed
|       |   group_sampleno.tif
|       |   ...
└─ registered
    |   group_sampleno.tif
    |   mask_overlay_group_sampleno.svg
    |   ...
```

## Example data
Example data to run this workflow can be found on [Zenodo]().