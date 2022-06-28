#!/bin/bash

#SBATCH --job-name=sim
#SBATCH --output=job-output/additive_var_%A_%a.out
#SBATCH --error=job-output/additive_var_%A_%a.err
#SBATCH --array=1-100
#SBATCH --time=20:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --account=pi-andyd


module load python
source activate py38

# SLURM_ARRAY_TASK_ID=1

SEED=$((42 + $SLURM_ARRAY_TASK_ID))
PARAM="additive_var"

# Print the task id.
echo "Repeat:" $SLURM_ARRAY_TASK_ID
echo "Seed:" $SEED

for Ktrue in 2 3 4 10
do
    PREFIX="${PARAM}_k${Ktrue}_self"
	if [ ! -d $PREFIX ]; then
        mkdir $PREFIX 
    fi

    echo "Ktrue =" $Ktrue
    python generate_data.py --self $Ktrue $SLURM_ARRAY_TASK_ID $SEED $PARAM

    echo "identified model"
    python benchmark.py 2 $Ktrue $SLURM_ARRAY_TASK_ID $PARAM coordinated
    echo "additive model"
    python benchmark.py 0 $Ktrue $SLURM_ARRAY_TASK_ID $PARAM additive
    echo "uncoordinated model"
    python benchmark.py 0 $Ktrue $SLURM_ARRAY_TASK_ID $PARAM uncoordinated
    echo "uncoordinated model random effects"
    python benchmark.py 0 $Ktrue $SLURM_ARRAY_TASK_ID $PARAM uncoordinated_random

    for Kmodel in 2 3 4
    do
        echo "coordinated model, Kmodel " $Kmodel
        python benchmark.py $Kmodel $Ktrue $SLURM_ARRAY_TASK_ID $PARAM coordinated_self
    done
done
echo "done"


