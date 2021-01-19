#!/bin/bash

#SBATCH --partition=              # because I can only use the fat node
#SBATCH -n                          # number of cores
#SBATCH --time=01-00:00:00           # Max time for task. Format is DD-HH:MM:SS
#SBATCH -o slurm.spz.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.spz.%j.err          # STDERR (%j = JobId)
#SBATCH --mail-type=ALL              # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=bjoshi5@jhu.edu  # send-to address

module restore    # Always restore/purge modules to ensure a consistent environment

module load python/3.7-anaconda
conda activate /home-1/bjoshi5@jhu.edu/data/conda_envs/romanslitless
