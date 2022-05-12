#!/bin/bash

#PBS -N epistasis_ukb
#PBS -S /bin/bash
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=15gb
#PBS -o $HOME/epistasis_ukb$PBS_JOBID.out
#PBS -e $HOME/epistasis_ukb$PBS_JOBID.err

source /gpfs/data/ukb-share/dahl/jerome/my-base-env/bin/activate /gpfs/data/ukb-share/dahl/jerome/my-base-env/envs/epistasis

cd /gpfs/data/ukb-share/dahl/jerome/epistasis-factorization/ukb

echo "Fitting with $pheno_file $model_file $self_interact $anchor $n_restarts $seed $k $permute"

python kfold_fit.py $pheno_file $model_file $self_interact $anchor $n_restarts $seed $k $permute
