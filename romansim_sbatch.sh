#!/bin/bash

#SBATCH --partition=parallel              # Partiton requested

#SBATCH -c 32                           # number of cores
#SBATCH --time=01-00:00:00              # Max time for task. Format is DD-HH:MM:SS
#SBATCH --mail-type=ALL                 # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=bjoshi5@jhu.edu     # send-to address
#SBATCH -o slurm.romansim.%j.out        # STDOUT (%j = JobId)
#SBATCH -e slurm.romansim.%j.err        # STDERR (%j = JobId)

# Typically need the large memory node because pyLINEAR is memory intensive
# Use --partition=express while testing code

echo "Starting SLURM script"

set -e

module restore

### source $HOME/.bashrc

ml anaconda
conda activate $HOME/code/conda_envs/romanslitless

echo "Starting python code from sbatch script..."
python run_pylinear.py
