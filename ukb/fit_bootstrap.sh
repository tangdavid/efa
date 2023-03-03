#!/bin/bash

#PBS -S /bin/bash
#PBS -l walltime=36:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=8gb

source /gpfs/data/ukb-share/dahl/jerome/my-base-env/bin/activate /gpfs/data/ukb-share/dahl/jerome/my-base-env/envs/epistasis

cd /gpfs/data/ukb-share/dahl/davidtang/epistasis-factorization

#for i in $(seq 0 9); do
#    echo "Bootstrap Number ${i}"
cmd=( python ./fit.py )  

if [[ "$permute" == "True" ]]; then
    cmd+=( --permute )                   
    cmd+=( --permute-seed $seed )
    cmd+=( --out ./ukb/pkl/bootstrap/${pheno}_${snps}_top${n_snps}/bootstrap_perm_${p}.pkl )
elif [[ "$rint" == "True" ]]; then
    cmd+=( --rint )
    cmd+=( --out ./ukb/pkl/bootstrap/${pheno}_${snps}_top${n_snps}/bootstrap_rint_${p}.pkl )
else
    cmd+=( --out ./ukb/pkl/bootstrap/${pheno}_${snps}_top${n_snps}/bootstrap_${p}.pkl )
fi

if [[ "$sink" == "True" ]]; then
    cmd+=( --sink )
fi

cmd+=( --init-noise $init_noise )
cmd+=( --restarts $n_restarts )
cmd+=( --algo $algo )
cmd+=( --bootstrap )
cmd+=( $pheno_file )

"${cmd[@]}"
#done
