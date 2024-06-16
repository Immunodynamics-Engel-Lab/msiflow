FROM python:3.12

ENV CORES=all
ENV WORKFLOW='msi_segmentation_flow'

RUN python3.12 -m pip install --upgrade pip

RUN apt-get update -y

WORKDIR /home/user/

COPY . /home/user/msiflow

WORKDIR /home/user/msiflow

ENV PYTHONPATH "${PYTHONPATH}:/home/user/msiflow"

# For fixing ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt install -y libgl1-mesa-glx

# install all python packages
RUN python3.12 -m pip install -r /home/user/msiflow/requirements.txt

CMD snakemake --cores $CORES --snakefile /home/user/msiflow/$WORKFLOW/Snakefile