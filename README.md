<p align="center">
  <img src="imgs/msiFlow_logo.svg" width=500/>
</p>

<h2 align="center">
automated workflows for reproducible and scalable multimodal mass spectrometry imaging and immunofluorescence microscopy data processing and analysis
</h2>

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

msiFlow can be run via **Docker** or **locally** by using a **graphical user interface** (gui) or the **command-line interface** (cli):
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
  - a description of all parameters is provided in the [wiki]()
- where and what **output** files are generated
- where to find **example data** to test the workflow

## Example data
Generally all example data to test the workflows can be found on [Zenodo]().


