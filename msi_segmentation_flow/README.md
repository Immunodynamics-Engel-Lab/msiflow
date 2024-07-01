# Workflow for MSI segmentation
This snakemake workflow performs single-sample or multi-sample segmentation of MSI data (provided as imzML files).
It includes state-of-the art dimensionality reduction methods and clustering algorithms to perform unsupervised 
segmentation. See workflow DAG [here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/msi_segmentation_flow/dag.pdf).

## Installation
Please see installation instructions [here](https://github.com/Immunodynamics-Engel-Lab/msiflow).

## Run
Please see [here](https://github.com/Immunodynamics-Engel-Lab/msiflow) how to run the **graphical-user interface** of msiFlow.
The following provides instructions on how to run the workflow via the **command-line interface**.

To run this workflow via **Docker** follow these instructions:
  - start Docker
  - in a terminal run `docker run -v <path-to-data-and-config>:/home/user/msiflow/data -e 'WORKFLOW=msi_segmentation_flow' -e 'CORES=<number-of-cores>' phispa1812/msiflow`

To run this workflow **locally** follow these instructions:
- in a terminal navigate to the root directory of msiFlow
- if you have installed the packages in a conda environment, activate the environment via `conda activate msiflow_env`
- run `snakemake --snakefile msi_segmentation_flow/Snakefile --cores <number-of-cores> --configfile <path-to-config>`

In the commands above
- enter the path of your data folder for `<path-to-data-and-config>`. See below how to structure and 
name your files in your data folder for successful completion of msiFlow.
- enter the max. number of cores to be used by msiFlow for `<number-of-cores>`. To provide all cores type *all*.
- enter the path of your configuration file for `<path-to-config>` when using the **local** version. 

## Input data
#### Data folder
Your data folder must contain the following directories and files:
- directory named *msi* containing imzML files named *group_sampleno.imzML*
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
    |   group_sampleno.imzML
    |   ...
```

## Output
The workflow outputs various visualisations of the segmentation and dimensionality reduction. You will find the results 
in a subfolder of the input directory containing the imzML files named *segmented*.

**Example of multisample segmentation:**
```
data
|   config.yaml
└─ msi
    |   group_sampleno.imzML
    |   ...
    └─ segmented
        |   class_group_sampleno.tif
        |   ... 
        |   clusters.png
        |   combined_2D_umap_colored_by_group.png
        |   combined_2D_umap_colored_by_sample.png
        |   umap_data.csv
        |   umap_model.sav
        |   umap_sample_group_sampleno.png
        |   ...
        |   group_2D_umap_clusters.png
        |   group_2D_umap_colored_by_group.png
        |   group_2D_umap_colored_by_sample.png
        |   ...
        |   group_sampleno.png
        |   ...
```

## Example data
Example data to run this workflow can be found on [Zenodo](https://doi.org/10.5281/zenodo.11913042) in *msi_segmentation.zip*.