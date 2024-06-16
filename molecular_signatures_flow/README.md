# Workflow to reveal molecular signatures
This snakemake workflow extracts molecular signatures from MSI data (provided as imzML files) 
for defined regions (provided as binary tif images/ image stacks)
by combining ML-based classification, SHAP values and similarity measures (Pearson and cosine) 
See workflow DAG 
[here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/molecular_signatures_flow/dag.pdf).

## Installation
please see installation instructions [here](https://github.com/Immunodynamics-Engel-Lab/msiflow).

## Run
To run this workflow via **Docker** follow these instructions:
  - start Docker
  - in a terminal run `docker run -v <path-to-data-and-config>:/home/user/msiflow/data -e 'WORKFLOW=molecular_signatures_flow' -e 'CORES=<number-of-cores>' phispa1812/msiflow`

To run this workflow **locally** follow these instructions:
- in a terminal navigate to the root directory of msiFlow
- if you have installed the packages in a conda environment, activate the environment via `conda activate msiflow_env`
- run `snakemake --snakefile molecular_signatures_flow/Snakefile --cores <number-of-cores> --configfile <path-to-config>`

In the commands above
- enter the path of your data folder for `<path-to-data-and-config>`. See below how to structure and 
name your files in your data folder for successful completion of msiFlow.
- enter the max. number of cores to be used by msiFlow for `<number-of-cores>`. To provide all cores type *all*.
- enter the path of your configuration file for `<path-to-config>` when using the **local** version. 

## Input data
#### Data folder
Your data folder must contain the following directories and files:
- directory named *msi* containing imzML files named *group_sampleno.imzML*
- directory named *bin_imgs* containing binary images named *class_group_sampleno.tif* or directory with image stacks named *group_sampleno.tif*.
- optional LipidMaps file named *annotation.tsv* containing potential lipid matches to measured m/z values
- optional file with UMAP embedding (generated with `msi_segmentation_flow` or `molecular_heterogeneity_flow`) named *umap_data.csv*  
 to save UMAPs with intensity distribution top m/z
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
|   annotation.tsv
|   umap_data.csv
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
- subdirectory of *msi* named *meas_pixels* containing binary TIF images of measured MSI pixels
- subdirectory of *bin_imgs* named *reduced_to_meas_pixels* containing binary images reduced to measured MSI pixels
- subdirectory of *reduced_to_meas_pixels* named *class* containing binary TIF images with positive and negative pixels for class
- subdirectory of *class* named *spatial_similarity* containing m/z ranking based on 
cosine and Pearson's correlation for each sample and the overall (mean)
- subdirectory of *class* named *<ML-model_balancing_method>* 
containing feature importance ranking of m/z values and SHAP summary
- subdirectory of *class* named *combined_rankings* containing the top features as csv and feature importance ranking 
  colour-coded according to the Pearson's correlation

**Example:**
```
data
|   config.yaml
|   annotation.tsv
|   umap_data.csv
└─ msi
|   |   group_samplono.ibd
|   |   group_sampleno.imzML
|   |   ...
|   └─ meas_pixels
|       |   group_sampleno.tif
|       |   ... 
└─ bin_imgs
        |   class_group_sampleno.tif
        |   ...
        └─ reduced_to_meas_pixels
            |   class_group_sampleno.tif
            |   ...
            └─ model_balancingmethod
            |   |   model_feature_importance.svg
            |   |   model_shap_summary.svg
            |   |   ...
            └─ ...
            └─ class
                | neg_group_sampleno.tif
                | pos_group_sampleno.tif
                | ...
                └─ spatial_similarity
                |   | barplot_pearson_corr.svg
                |   | barplot_cosine_sim.svg
                |   | cosine_group_sampleno.csv
                |   | pearson_group_sampleno.csv
                |   | overall_spatial_ranking_pearson.csv
                |   | overall_spatial_ranking_cosine.csv
                |   | top_mz_group_sampleno.tif
                |   | venn_similarity_measures.svg
                |   | violinplot_cosine_sim.svg
                |   | violinplot_pearson_corr.svg
                |   | ...
                └─ model_balancingmethod
                |   └─ ion_imgs
                |   |   └─ group_sampleno
                |   |   |   |   group_sampleno_mz.tif
                |   |   |   |   ...
                |   |   └─  ...
                |   └─ ion_umaps
                |   |   |   mz_umap.png
                |   |   |   ...
                |   |   model_feature_importance.csv 
                |   |   model_feature_importance.svg
                |   |   model_shap_summary.svg
                |   |   model.json
                └─ combined_rankings
                    |   annot_top_features.svg
                    |   top_features.csv
                    |   top_features.svg
```
## Example data
Example data to run this workflow can be found on [Zenodo]().