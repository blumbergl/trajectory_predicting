#!/bin/sh

#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --output=./slurm_log/merge_csv-%j.out

echo 'executing phiz.sh'
date

source activate decoding
time python3 phizzyForReal.py

echo 'finished!'
date