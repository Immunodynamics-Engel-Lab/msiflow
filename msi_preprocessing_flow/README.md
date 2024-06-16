# Workflow for MSI data pre-processing
This snakemake workflow imports raw MSI files (.d and .imzML), processes all files in parallel, and outputs the 
processed data in imzML format along with quality control visualisations. The workflow contains the follwing steps:
1. spectral smoothing
2. peak picking
3. peak alignment
4. matrix removal
5. peak filtering
6. normalisation
7. outlier removal
8. de-isotoping 

After all steps an endogenous/tissue-origin mono-isotopic peak list is generated. See workflow DAG 
[here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/msi_preprocessing_flow/dag.pdf).

## Installation
please see installation instructions [here](https://github.com/Immunodynamics-Engel-Lab/msiflow).

## Run
To run this workflow via **Docker** follow these instructions:
  - start Docker
  - in a terminal run `docker run -v <path-to-data-and-config>:/home/user/msiflow/data -e 'WORKFLOW=msi_preprocessing_flow' -e 'CORES=<number-of-cores>' phispa1812/msiflow`

To run this workflow **locally** follow these instructions:
- in a terminal navigate to the root directory of msiFlow
- if you have installed the packages in a conda environment, activate the environment via `conda activate msiflow_env`
- run `snakemake --snakefile msi_preprocessing_flow/Snakefile --cores <number-of-cores> --configfile <path-to-config>`

In the commands above
- enter the path of your data folder for `<path-to-data-and-config>`. See below how to structure and 
name your files in your data folder for successful completion of msiFlow.
- enter the max. number of cores to be used by msiFlow for `<number-of-cores>`. To provide all cores type *all*.
- enter the path of your configuration file for `<path-to-config>` when using the **local** version. 

## Input data
#### Data folder
Your data folder must contain the following directories and files:
- directory named *msi* containing imzML files or .d directories named *group_sampleno.imzML/.d*
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
└─ msi
    └─ group_sampleno.d
    └─ ...
```

## Output
The workflow creates sub-directories for each pre-processing step containing the processed data in imzML format along 
with quality control visualisations.

**Example:**
```
data
|   config.yaml
└─ msi
    └─ group_sampleno.d
    └─ peakpicking_*snr
        |   group_sampleno.ibd
        |   group_sampleno.imzML
        |   ...
        └─ alignment_*pxperc
            |   group_sampleno.ibd
            |   group_sampleno.imzML
            |   ...
            └─ umap_*neighb_hdbscan_*ms_*mcs
            |   └─ binary_imgs
            |   |   |   group_sampleno_cluster*.tif  
            |   |   |   group_sampleno_matrix_cluster.tif
            |   |   |   ...
            |   |   group_sampleno_border_pixels.svg
            |   |   group_sampleno_cluster_image.svg
            |   |   group_sampleno_cluster_scatterplot.png
            |   |   group_sampleno_matrix_pixels.csv
            |   |   ...
            └─ matrix_removal
                |   group_sampleno.ibd
                |   group_sampleno.imzML
                |   group_sampleno_extended_matrix_img.tif
                |   group_sampleno_postproc_matrix_img.tif
                |   group_sampleno_sc_distribution.svg
                |   group_sampleno_sc.csv
                |   overall_spatial_coherence.csv
                |   peaks_above_*sc.npy
                |   ...
                └─ quality_control
                |   |   group_sampleno_cluster_pixel_perc.svg
                |   |   group_sampleno_corr.svg
                |   |   group_sampleno_matrix_cluster_corr.svg
                |   |   group_sampleno_matrix_cluster.tif
                |   |   tissue_clusters.tif
                └─ *sc_filtered
                    |   group_sampleno.ibd   
                    |   group_sampleno.imzML
                    |   ...
                    └─ intranorm_*
                        |   group_sampleno.ibd   
                        |   group_sampleno.imzML 
                        |   ...  
                        └─ quality_control
                        |   |   group_sampleno_(x,y)_after_norm.svg
                        |   |   group_sampleno_(x,y)_before_norm.svg
                        |   |   group_sampleno_scfactors.svg
                        |   |   ...
                        └─ internorm_*
                            |   group_sampleno.ibd   
                            |   group_sampleno.imzML 
                            |   ...   
                            └─ umap_*neighb_hdbscan_*ms_*mcs
                            |   |   class_group_sampleno_image.tif
                            |   |   clusters.png
                            |   |   group_sampleno.png
                            |   |   umap_data.csv
                            |   |   ...
                            └─ outlier_removal
                                |   group_sampleno.ibd   
                                |   group_sampleno.imzML  
                                |   ...
                                └─ quality_control
                                |   |   class_sample_pixels.svg
                                |   |   barplot.svg
                                |   |   ...
                                └─ deisotoped
                                    |   group_sampleno.ibd   
                                    |   group_sampleno.imzML   
                                    |   ...
```

## Example data
Example data to run this workflow can be found on [Zenodo]().