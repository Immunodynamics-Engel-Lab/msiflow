#!/bin/bash

#SBATCH --job-name=msiflow-preprocessing-test
#SBATCH --output=msiflow-preprocessing-test-output-%j.log
#SBATCH --partition=normal
#SBATCH -A uni
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=90G
#SBATCH --time=24:00:00

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Navigate to software directory
cd /home/s/spangenp/software/msiflow-1.0.1/

# Load the modules
ml palma/2023a
ml foss LLVM/14.0.6-llvmlite # Native dependency of Numba
ml uv

# create Python environment
uv venv --python 3.12
source .venv/bin/activate
uv pip install --no-cache-dir -r requirements.txt
uv pip install koyo==0.2.10

# run msiFlow preprocessing
srun snakemake --snakefile msi_preprocessing_flow/Snakefile --cores $SLURM_CPUS_PER_TASK --configfile /scratch/tmp/spangenp/msi_preprocessing_test/config.yaml --resources mem_mb=90000 --rerun-incomplete
