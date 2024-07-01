# Workflow to unravel molecular heterogeneity
This snakemake workflow identifies molecular heterogeneity from MSI data (provided as imzML files)
of specific regions (provided as binary tif images) by performing UMAP and HDBSCAN clustering.
The workflow `molecular_signatures_flow` can be subsequently applied 
to extract molecular signatures of identified clusters. 
See workflow DAG 
[here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/molecular_heterogeneity_flow/dag.pdf).

## Installation
Please see installation instructions [here](https://github.com/Immunodynamics-Engel-Lab/msiflow).

## Run
Please see [here](https://github.com/Immunodynamics-Engel-Lab/msiflow) how to run the **graphical-user interface** of msiFlow.
The following provides instructions on how to run the workflow via the **command-line interface**.

To run this workflow via **Docker** follow these instructions:
  - start Docker
  - in a terminal run `docker run -v <path-to-data-and-config>:/home/user/msiflow/data -e 'WORKFLOW=molecular_heterogeneity_flow' -e 'CORES=<number-of-cores>' phispa1812/msiflow`

To run this workflow **locally** follow these instructions:
- in a terminal navigate to the root directory of msiFlow
- if you have installed the packages in a conda environment, activate the environment via `conda activate msiflow_env`
- run `snakemake --snakefile molecular_heterogeneity_flow/Snakefile --cores <number-of-cores> --configfile <path-to-config>`

In the commands above
- enter the path of your data folder for `<path-to-data-and-config>`. See below how to structure and 
name your files in your data folder for successful completion of msiFlow.
- enter the max. number of cores to be used by msiFlow for `<number-of-cores>`. To provide all cores type *all*.
- enter the path of your configuration file for `<path-to-config>` when using the **local** version. 

## Input data
#### Data folder
Your data folder must contain the following directories and files:
- directory named *msi* containing imzML files named *group_sampleno.imzML*
- directory named *bin_imgs* containing ROIs as binary images named *roi_group_sampleno.tif*
- optional file of trained UMAP model (required for reproducing results as UMAP is stochastic) named *umap_model.sav*
- *config.yaml* when using the **Docker** version of msiFlow with the **command-line interface** 

#### Configuration file
When using the **command-line interface** you must provide a **configuration file** named *config.yaml* to run this workflow. All parameters used by the workflow are defined in
this configuration file. See the [wiki]() for a description of all parameters. An example configuration file can be
found [here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/msi_segmentation_flow/data/config.yaml).

When using the **Docker** version of msiFlow the *config.yaml* must be provided in the data folder. 

For the **local** version of msiFlow the path of the configuration file can be specified in the command.

**Example:**
```
data
|   config.yaml
└─ msi
|   |   group_samplono.ibd
|   |   group_sampleno.imzML
|   |   ...
└─ bin_imgs
    |   class_group_sampleno.tif
    |   ...
```

## Output
The workflow outputs the following directories and files:
- subdirectory of *bin_imgs* named *class* containing imzML files reduced to pixels from provided binary images
- subdirectory of *class* named *umap* containing clustering results from UMAP and HDBSCAN

**Example:**
```
data
|   config.yaml
└─ msi
|   |   group_samplono.ibd
|   |   group_sampleno.imzML
|   |   ...
└─ bin_imgs
    |   class_group_sampleno.tif
    |   ...
    └─ class
        |   class_group_sampleno.tif
        |   ...
        └─ imzML
        |   |   group_sampleno.ibd
        |   |   group_sampleno.imzML
        |   |   ...
        └─ umap_*neighb_hdbscan_*ms_*mcs
            |   class_group_sampleno.tif
            |   clusters.png
            |   group_sampleno.png
            |   ...
```

## Example data
Example data to run this workflow can be found on [Zenodo](https://doi.org/10.5281/zenodo.11913042) in *ly6g_heterogeneity.zip*.