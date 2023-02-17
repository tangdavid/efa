#!/bin/bash

source /gpfs/data/ukb-share/dahl/jerome/my-base-env/bin/activate /gpfs/data/ukb-share/dahl/jerome/my-base-env/envs/epistasis

cd /gpfs/data/ukb-share/dahl/davidtang/epistasis-factorization/ukb

PHENO_DIR="/gpfs/data/ukb-share/dahl/jerome/extracted_phenotypes/epistasis_pheno"

export pheno="male_testosterone"
export snps="sinarm-maletest-range"
export n_snps="100"
export seed=42
export folds=2
export n_restarts=20
export rint='False'
export sink='False'
export algo='coord'
export self_interact='False'
export anchor='False'
export pheno_file="${PHENO_DIR}/${pheno}_${snps}_top${n_snps}.raw"

n_permutations=1000

echo "Submitting unpermuted..."
model_file="pkl/${pheno}_${snps}_top${n_snps}_baseline.pkl"
qsub -V -v permute=False,model_file=${model_file} fit_ukb.sh \
    -o ./job-out/${pheno}.out \
    -e ./job-out/${pheno}.err \
    -N ${pheno}
sleep 0.5

mkdir -p "pkl/permutations/${pheno}_${snps}_top${n_snps}_baseline"

for p in $(seq 1 $n_permutations); do
    model_file="pkl/permutations/${pheno}_${snps}_top${n_snps}_baseline/perm${p}.pkl"
    echo "Submitting permutation" $p
    qsub -V -v permute=True,p=${p},model_file=${model_file} fit_ukb.sh \
        -o ./job-out/${pheno}_${p}.out \
        -e ./job-out/${pheno}_${p}.err \
        -N ${pheno}_${p}
    sleep 0.5
done
