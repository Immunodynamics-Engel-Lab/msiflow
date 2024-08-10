# Workflow to identify molecular changes in regions between groups
This snakemake workflow is designed to identify and compare molecular changes in different ROIs (e.g. tissue regions) between 
two groups by applying statistical analysis of MSI data. See workflow DAG 
[here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/region_group_analysis_flow/dag.pdf).

## Installation
Please see installation instructions [here](https://github.com/Immunodynamics-Engel-Lab/msiflow).

## Run
Please see [here](https://github.com/Immunodynamics-Engel-Lab/msiflow) how to run the **graphical-user interface** of msiFlow.
The following provides instructions on how to run the workflow via the **command-line interface**.

To run this workflow via **Docker** follow these instructions:
  - start Docker
  - in a terminal run `docker run -v <path-to-data-and-config>:/home/user/msiflow/data -e 'WORKFLOW=region_group_analysis_flow' -e 'CORES=<number-of-cores>' phispa1812/msiflow`

To run this workflow **locally** follow these instructions:
- in a terminal navigate to the root directory of msiFlow
- if you have installed the packages in a conda environment, activate the environment via `conda activate msiflow_env`
- run `snakemake --snakefile region_group_analysis_flow/Snakefile --cores <number-of-cores> --configfile <path-to-config>`

In the commands above
- enter the path of your data folder for `<path-to-data-and-config>`. See below how to structure and 
name your files in your data folder for successful completion of msiFlow.
- enter the max. number of cores to be used by msiFlow for `<number-of-cores>`. To provide all cores type *all*.
- enter the path of your configuration file for `<path-to-config>` when using the **local** version. 

## Input data
#### Data folder
Your data folder must contain the following directories and files:
- directory named *msi* containing imzML files named *group_sampleno.imzML*
- directory named *bin_imgs* containing image files of the binary regions named *class_group_sampleno.tif*
- *config.yaml* when using the **Docker** version of msiFlow with the **command-line interface** 

#### Configuration file
When using the **command-line interface** you must provide a **configuration file** named *config.yaml* to run this workflow. All parameters used by the workflow are defined in
this configuration file. See the [wiki](https://github.com/Immunodynamics-Engel-Lab/msiflow/wiki/Parameters#region-group-analysis-workflow) for a description of all parameters. An example configuration file can be
found [here](https://github.com/Immunodynamics-Engel-Lab/msiflow/blob/main/msi_segmentation_flow/data/config.yaml).

When using the **Docker** version of msiFlow the *config.yaml* must be provided in the data folder. 

For the **local** version of msiFlow the path of the configuration file can be specified in the command.

**Example:**
```
data
|   config.yaml
|   annotation.tsv
└─ msi
|   |   group_sampleno.ibd
|   |   group_sampleno.imzML
|   |   ...
└─ bin_imgs
    |   class_group_sampleno.tif
    |   ...
```

## Output
The workflow outputs the following directories and files:
- directory *class* containing imzML files reduced to the pixels of class from binary image
- directory *class/summarized/FC<FC_thr>* containing volcano plot, heatmap of regulated m/z values, 
pie chart of regulated lipids (if annotated), csv files with regulated m/z values
- directory *combined_FC<FC_thr>_<down/up>regulated* containing Venn diagram with intersecting regulation 
between regions (if 2-3 regions provided) and csv files with intersecting and unique regulated m/z for every class

**Example:**
```
data
|   config.yaml
|   annotation.tsv
└─ msi
|   |   group_sampleno.ibd
|   |   group_sampleno.imzML
|   |   ...
└─ bin_imgs
|   |   class_group_sampleno.tif
|   |   ...
└─ class
|   |   group_sampleno.ibd
|   |   group_sampleno.imzML 
|   |   ...
|   └─ summarized
|   |   |   annot_summarized.csv
|   |   └─ quality_control
|   |   |   |   corr_heatmap.svg
|   |   |   |   samples_boxplot.svg
|   |   |   |   ...
|   |   └─ FC*
|   |       |   annot_group1_group2_analysis.csv
|   |       |   annot_group1_group2_regulated.csv
|   |       |   annot_group1_group2_regulated_heatmap.svg
|   |       |   annot_group1_group2_regulated_piechart.svg
|   |       |   annot_volcano_plot.svg
|   |       |   ...
└─ combined_FC*_*regulated
|   |   common_molecules.csv
|   |   class_specific_molecules.csv
|   |   venn_diagram.svg
|   |   ...   
└─ ...
```

## Example data
Example data to run this workflow can be found on [Zenodo](https://doi.org/10.5281/zenodo.11913042) in *region_group_analysis.zip*.