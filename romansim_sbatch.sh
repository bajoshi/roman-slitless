#!/bin/bash

#SBATCH --partition=shared              # Partiton requested
#SBATCH -c 10                           # number of cores
#SBATCH --time=00-00:05:00              # Max time for task. Format is DD-HH:MM:SS
#SBATCH --mail-type=ALL                 # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=bjoshi5@jhu.edu     # send-to address

# Typically need the large memory node because pyLINEAR is memory intensive
# Use --partition=express while testing code

set -e

module restore    # Always restore/purge modules to ensure a consistent environment

module load python/3.7-anaconda
conda activate /home-1/bjoshi5@jhu.edu/data/conda_envs/romanslitless

print("Starting python code from sbatch script...")
python run_pylinear.py