#!/bin/bash

#SBATCH --job-name=LD
#SBATCH --output=job-output/LD_%A_%a.out
#SBATCH --error=job-output/LD_%A_%a.err
#SBATCH --array=1-1000
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --account=pi-andyd


module load python
source activate py38

# SLURM_ARRAY_TASK_ID=1

SEED=$((42 + $SLURM_ARRAY_TASK_ID))
NAME="LD"

# Print the task id.
echo "Repeat:" $SLURM_ARRAY_TASK_ID
echo "Seed:" $SEED

    PREFIX="${NAME}_perm"
	if [ ! -d $PREFIX ]; then
        mkdir $PREFIX 
        mkdir "${PREFIX}/null"
    fi

    if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
        bash generate_perm_null.sbatch $NAME
    else 
       sleep 2m 
    fi

    echo 'generating splits...'
    python generate_perm.py $SLURM_ARRAY_TASK_ID $SEED $NAME

    echo 'fitting on unpermutated data...'
    python benchmark_perm.py $SLURM_ARRAY_TASK_ID $NAME
    rm ${PREFIX}/training_${SLURM_ARRAY_TASK_ID}.pkl
    rm ${PREFIX}/validation_${SLURM_ARRAY_TASK_ID}.pkl

    echo 'fitting on permuted data...'
    python benchmark_perm.py --null $SLURM_ARRAY_TASK_ID $NAME
    rm ${PREFIX}/null/training_${SLURM_ARRAY_TASK_ID}.pkl
    rm ${PREFIX}/null/validation_${SLURM_ARRAY_TASK_ID}.pkl

echo "done"


