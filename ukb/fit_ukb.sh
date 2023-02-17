#!/bin/bash

#PBS -S /bin/bash
#PBS -l mem=8gb
#PBS -l walltime=24:00:00 
#PBS -l nodes=1:ppn=4

##msub -S /bin/bash
##msub -l pmem=15gb
##msub -l walltime=12:00:00 
##msub -l nodes=1:ppn=4
##msub -e ./job-out/${pheno}%I.err
##msub -o ./job-out/${pheno}%I.out

source /gpfs/data/ukb-share/dahl/jerome/my-base-env/bin/activate /gpfs/data/ukb-share/dahl/jerome/my-base-env/envs/epistasis

cd /gpfs/data/ukb-share/dahl/davidtang/epistasis-factorization/ukb

echo "Fitting with $pheno_file $model_file $self_interact $anchor $n_restarts $seed $folds $permute $rint $sink $init_noise $algo"

python kfold_fit.py \
    $pheno_file \
    $model_file \
    $self_interact \
    $anchor \
    $n_restarts \
    $seed \
    $folds \
    $permute \
    $rint \
    $sink \
    $init_noise \
    $algo
