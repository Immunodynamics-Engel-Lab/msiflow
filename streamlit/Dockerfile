FROM python:3.12

ENV CORES=all
ENV WORKFLOW=msi_segmentation_flow

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /home/user/

COPY . /home/user/msiflow

WORKDIR /home/user/msiflow

ENV PYTHONPATH="${PYTHONPATH}:/home/user/msiflow"

RUN python -m pip install -r /home/user/msiflow/streamlit/requirements.txt
RUN python -m pip install koyo==0.2.10

CMD streamlit run /home/user/msiflow/streamlit/home.py