#!/bin/bash

source /gpfs/data/ukb-share/dahl/jerome/my-base-env/bin/activate /gpfs/data/ukb-share/dahl/jerome/my-base-env/envs/epistasis

cd /gpfs/data/ukb-share/dahl/davidtang/epistasis-factorization/ukb

PHENO_DIR="/gpfs/data/ukb-share/dahl/jerome/extracted_phenotypes/epistasis_pheno"

param_row=$1

params=`cat params.csv | head -n $param_row | tail -n1`

export pheno=`echo "$params" | cut -d',' -f1`
export snps=`echo "$params" | cut -d',' -f2`
export permute=`echo "$params" | cut -d',' -f3`
export rint=`echo "$params" | cut -d',' -f4`

#export permute='False'
#export rint='False'

export n_snps="100"
export n_restarts=1
export init_noise=0.1
export algo='coord'
export sink='False'
export seed=42
export pheno_file="${PHENO_DIR}/${pheno}_${snps}_top${n_snps}.raw"

n_bootstrap=499
mkdir -p "pkl/bootstrap/${pheno}_${snps}_top${n_snps}"


for p in $(seq 0 $n_bootstrap); do
    if [[ $permute == 'True' ]]; then
        name=${pheno}_bootstrap_perm_${p}
    elif [[ $rint == 'True' ]]; then
        name=${pheno}_bootstrap_rint_${p}
    else
        name=${pheno}_bootstrap_${p}
    fi
    echo "Submitting" $name
    qsub -V -v p=${p} fit_bootstrap.sh \
        -o ./job-out/${name}.out \
        -e ./job-out/${name}.err \
        -N ${name}
    sleep 0.1
done
