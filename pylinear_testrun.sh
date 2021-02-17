#!/bin/bash

#SBATCH --partition=parallel              # Partiton requested

#SBATCH -c 16                           # number of cores
#SBATCH --time=00-02:00:00              # Max time for task. Format is DD-HH:MM:SS
#SBATCH --mail-type=ALL                 # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=bjoshi5@jhu.edu     # send-to address
#SBATCH -o slurm.romansim.%j.out        # STDOUT (%j = JobId)
#SBATCH -e slurm.romansim.%j.err        # STDERR (%j = JobId)

# Typically need the large memory node because pyLINEAR is memory intensive
# Use --partition=express while testing code

echo "Starting SLURM script to test pylinear versions"

set -e

module restore

### source $HOME/.bashrc

ml anaconda
conda activate $HOME/code/conda_envs/romanslitless

echo "Starting pylinear run on a few sources from sbatch script..."
python pylinear_few_sources_testrun.py
