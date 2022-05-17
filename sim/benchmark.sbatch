#!/bin/bash

#SBATCH --job-name=additive_var
#SBATCH --output=job-output/additive_var_%A_%a.out
#SBATCH --error=job-output/additive_var_%A_%a.err
#SBATCH --array=1-100
#SBATCH --time=20:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --account=pi-andyd


module load python
source activate py38

#SLURM_ARRAY_TASK_ID=1

SEED=$((42 + $SLURM_ARRAY_TASK_ID))
PARAM="additive_var"

# Print the task id.
echo "Repeat:" $SLURM_ARRAY_TASK_ID
echo "Seed:" $SEED

for Ktrue in 2 3 4
do
    PREFIX="${PARAM}_k${Ktrue}"
	if [ ! -d $PREFIX ]; then
        mkdir $PREFIX 
    fi

    echo "Ktrue =" $Ktrue
    python generate_data.py $Ktrue $SLURM_ARRAY_TASK_ID $SEED $PARAM

    echo "identified model"
    python benchmark.py 2 $Ktrue $SLURM_ARRAY_TASK_ID $PARAM coordinated

    for Kmodel in 2 3 4
    do
        echo "Kmodel = " $Kmodel
        echo "coordinated model"
        python benchmark.py $Kmodel $Ktrue $SLURM_ARRAY_TASK_ID $PARAM coordinated_self
        echo "additive model"
        python benchmark.py $Kmodel $Ktrue $SLURM_ARRAY_TASK_ID $PARAM additive
        echo "uncoordinated model"
        python benchmark.py $Kmodel $Ktrue $SLURM_ARRAY_TASK_ID $PARAM uncoordinated
    done
done
echo "done"


