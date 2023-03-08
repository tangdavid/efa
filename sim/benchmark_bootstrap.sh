#!/bin/bash

#SBATCH --job-name=bootstrap_sim
#SBATCH --output=job-output/bootstrap_sim_%A_%a.out
#SBATCH --error=job-output/bootstrap_sim_%A_%a.err
#SBATCH --array=1-1000
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --account=pi-andyd

module load python
source activate py38

# SLURM_ARRAY_TASK_ID=1

echo "Repeat:" $SLURM_ARRAY_TASK_ID

echo "Null sim..."
python benchmark_bootstrap.py $SLURM_ARRAY_TASK_ID 'efa' 0

echo "Weak effects sim..."
python benchmark_bootstrap.py $SLURM_ARRAY_TASK_ID 'efa' 0.01

echo "String effects sim..."
python benchmark_bootstrap.py $SLURM_ARRAY_TASK_ID 'efa' 0.1

echo "Weak dominance sim..."
python benchmark_bootstrap.py $SLURM_ARRAY_TASK_ID 'dominance' 0.01

echo "Strong dominance sim..."
python benchmark_bootstrap.py $SLURM_ARRAY_TASK_ID 'dominance' 0.1

echo "LD sim..."
python benchmark_bootstrap.py $SLURM_ARRAY_TASK_ID 'LD' 0

echo "Done!"


