#!/bin/bash

#SBATCH --job-name=sim
#SBATCH --output=job-output/additive_%A.out
#SBATCH --error=job-output/additive_%A.err
#SBATCH --time=1:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --account=pi-andyd


module load python
source activate py38

SEED=42
NAME=$1

echo 'generating the null permutations...'
PREFIX="${NAME}_perm"
if [ ! -d $PREFIX ]; then
    mkdir $PREFIX 
    mkdir "${PREFIX}/null"
fi

python generate_perm.py --null 0 $SEED $NAME
echo "done"


