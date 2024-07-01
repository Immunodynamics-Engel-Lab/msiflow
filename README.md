<p align="center">
  <img src="imgs/msiFlow_logo.svg" width=500/>
</p>

<h2 align="center">
automated workflows for reproducible and scalable multimodal mass spectrometry imaging and immunofluorescence microscopy data processing and analysis
</h2>

[![License](https://img.shields.io/github/license/Immunodynamics-Engel-Lab/msiflow?color=green&style=flat)](https://github.com/Immunodynamics-Engel-Lab/msiflow/LICENSE)
&nbsp;
![GitHub top language](https://img.shields.io/github/languages/top/Immunodynamics-Engel-Lab/msiflow)
&nbsp;
[![Latest Release](https://img.shields.io/github/v/release/Immunodynamics-Engel-Lab/msiflow?style=flat)](https://github.com/Immunodynamics-Engel-Lab/msiflow)
&nbsp;
[![Release Date](https://img.shields.io/github/release-date/Immunodynamics-Engel-Lab/msiflow?style=flat)](https://github.com/Immunodynamics-Engel-Lab/msiflow/releases)
&nbsp;
[![Issues](https://img.shields.io/github/issues/Immunodynamics-Engel-Lab/msiflow?style=flat)](https://github.com/Immunodynamics-Engel-Lab/msiflow/issues)
&nbsp;
[![Pull Requests](https://img.shields.io/github/issues-pr/Immunodynamics-Engel-Lab/msiflow?style=flat)](https://github.com/Immunodynamics-Engel-Lab/msiflow/pulls)
&nbsp;
[![Commits](https://img.shields.io/github/commit-activity/m/Immunodynamics-Engel-Lab/msiflow?style=flat)](https://github.com/Immunodynamics-Engel-Lab/msiflow)
&nbsp;
[![Docker Downloads](https://img.shields.io/docker/pulls/phispa1812/msiflow_gui?style=flat)](https://hub.docker.com/r/phispa1812/msiflow_gui)
&nbsp;
[![GitHub Repo stars](https://img.shields.io/github/stars/Immunodynamics-Engel-Lab/msiflow)](https://github.com/Immunodynamics-Engel-Lab/msiflow) 
&nbsp;

msiFlow contains 7 snakemake workflows:

| task | description | workflow
| --- | --- | --- |
| pre-processing | generates a peak list from raw MALDI MSI data | `msi_preprocessing_flow` |
| registration | combines MSI & IFM including IFM pre-processing | `msi_if_registration_flow` |
| segmentation | dim. reduction and clustering of MSI data | `msi_segmentation_flow`|
| segmentation | thresholding-based segmentation of IFM markers | `if_segmentation_flow`|
| analysis & visualisation | molecular changes in different ROIs between two groups by statistical analysis | `region_group_analysis_flow` |
| analysis & visualisation | molecular signatures of ROIs by ML classification | `molecular_signatures_flow` |
| analysis & visualisation | molecular heterogeneity in ROIs by UMAP-based clustering | `molecular_heterogeneity_flow` |

msiFlow can be run via **Docker** or **locally** by using a graphical user interface (gui) or the command-line interface (cli):
1. The **Docker version** of msiFlow is intended for **easy-to-use** execution and does not require package installations.
2. The **local version** of msiFlow is intended for **development**.

## Installation

### Docker Version

To use the Docker version of msiFlow, follow these instructions:
1. install Docker for your operating system from [here](https://docs.docker.com/get-docker/).
2. start Docker
3. get the Docker image 
   1. from DockerHub by running `docker pull phispa1812/msiflow_<interface>` in a terminal. In the command enter `gui` for `<interface>` to install the **graphical user interface** or `cli` to install the **command-line interface**.
   2. or build the Docker image by navigating to the root directory of msiFlow in a terminal (e.g. via `cd msiFlow`) and running `docker build -f streamlit/Dockerfile -t msiflow_gui .` to install the **graphical user interface** or by running `docker build -t msiflow_cli .` to install the **command-line interface**.


### Local Version (without Docker)

To use the local version of msiFlow, follow these instructions:
1. download or clone this repository if you have Git installed by running `git clone https://github.com/Immunodynamics-Engel-Lab/msiflow.git` in a terminal
2. navigate to the root directory of msiFlow (e.g. via `cd msiFlow`) in a terminal
3. environment/package installation
   1. if you have Anaconda installed, create an environment of msiFlow by running `conda env create --file msiflow_env.yaml`. This creates an environemnt with all required packages at once.
   2. if you want to use pure Python, download all packages by running `pip install -r requirements.txt`

## Run msiFlow

### Docker Version

To run the Docker version of msiFlow, follow these instructions:
  - start Docker
  - start the **graphical user interface** by running `docker run -v <path-to-data>:/home/user/msiflow/data -p 8501:8501 phispa1812/msiflow_gui` in a terminal and navigating to `localhost:8501` in a browser (e.g. Firefox)
  - or execute a workflow via the **command-line interface** by running `docker run -v <path-to-data-and-config>:/home/user/msiflow/data -e 'WORKFLOW=<workflow>' -e 'CORES=<number-of-cores>' phispa1812/msiflow_cli` in a terminal

**Important note:** don't change the field *input path* when using the gui or the *data* parameter in the configuration file when using the cli! This parameter is only intended for the local version of msiFlow.
In the Docker version the data path is specified in the corresponding command by `<path-to-data>` or `<path-to-data-and-config>`. 
### Local Version

To run msiFlow locally, follow these instructions:
- in a terminal navigate to the root directory of msiFlow (e.g. via `cd msiFlow`)
- if you have installed the packages in a conda environment, activate the environment via `conda activate msiflow_env`
- start the **graphical user interface** by running `streamlit run streamlit/home.py` in a terminal and navigating to `localhost:8501` in a browser (e.g. Firefox)
- or execute a workflow via the **command-line interface** by running `snakemake --snakefile <workflow>/Snakefile --cores <number-of-cores> --configfile <path-to-config>` in a terminal

In the commands above
- enter the name of the workflow to be executed for `<workflow>`. The possible workflows to select are listed in the table above.
- enter the path of your data folder for `<path-to-data>`
- enter the path of your data folder which includes a configuration file for `<path-to-data-and-config>`
- enter the max. number of cores to be used by msiFlow for `<number-of-cores>`. To provide all cores type *all*.
- enter the path of your configuration file for `<path-to-config>` when using the **local** version. 

### Workflow-specific information

Each workflow directory contains a **README** with detailed information on 
- how to **run** the workflow via the command-line interface
- how to structure and name your **input data** files for successful execution of msiFlow
- how to set the **configuration** parameters 
  - example *config.yaml* is provided in each directory
  - a description of all parameters is provided in the [wiki](https://github.com/Immunodynamics-Engel-Lab/msiflow/wiki)
- where and what **output** files are generated
- where to find **example data** to test the workflow

## Example data
Generally all example data to test the workflows can be found on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11913042.svg)](https://doi.org/10.5281/zenodo.11913042)
## Correspondence
[**Prof. Dr. Daniel R. Engel**](mailto:danielrobert.engel@uk-essen.de): Department of Immunodynamics, Institute of Experimental Immunology and Imaging, University Hospital Essen, Essen, Germany

http://www.immunodynamics.de

## References
1. Veselkov, K., Sleeman, J., Claude, E. et al. BASIS: High-performance bioinformatics platform for processing of large-scale mass spectrometry imaging data in chemically augmented histology. Sci Rep 8, 4053 (2018). https://doi.org/10.1038/s41598-018-22499-z
2. Grélard, F., Legland, D., Fanuel, M. et al. Esmraldi: efficient methods for the fusion of mass spectrometry and magnetic resonance images. BMC Bioinformatics 22, 56 (2021). https://doi.org/10.1186/s12859-020-03954-z
3. imzy. imzy: A new reader/writer interface to imzML and other imaging mass spectrometry formats. GitHub repository (2022). https://github.com/vandeplaslab/imzy