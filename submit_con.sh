#!/bin/bash
#SBATCH --account=rrg-ovoznyy
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0-02:00
#SBATCH --job-name=conductivity
#SBATCH --output=__%j__.sto
#SBATCH --error=__%j__.err
#SBATCH --mail-user=orhan.kibrisli@utoronto.ca
#SBATCH --mail-type=END,FAIL,REQUEUE
# SBATCH --partition=debug

module load NiaEnv/2019b
module load intel/2019u4
source /gpfs/fs1/home/o/ovoznyy/orhan/general/bin/activate

python conductivity.py

# Define the results directory
RESULTS_DIR=__${SLURM_JOB_ID}__

# Ensure the results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
fi

# Copy the Python script to the results directory
cp conductivity.py "$RESULTS_DIR"/