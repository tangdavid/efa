#!/bin/bash

source /gpfs/data/ukb-share/dahl/jerome/my-base-env/bin/activate /gpfs/data/ukb-share/dahl/jerome/my-base-env/envs/epistasis

cd /gpfs/data/ukb-share/dahl/jerome/epistasis-factorization/ukb

PHENO_DIR="/gpfs/data/ukb-share/dahl/jerome/extracted_phenotypes/epistasis_pheno"
pheno_list="hba1c"
pheno_ids="30750"
pheno_baskets="29329"
snp_sets="xue-t2d"
n_snps="6;10;20;100"

n_snps_list=(${n_snps//;/ })
pheno_list_iter=(${pheno_list//;/ })
# run gen-plink-raw here, pass args -> use qsub -W depend=afterok:$pheno_job -v ...
echo "Submitting pheno job..."
pheno_job=$( qsub -v pheno_list=$pheno_list,pheno_ids=$pheno_ids,pheno_baskets=$pheno_baskets,snp_sets=$snp_sets,n_snps=$n_snps gen-plink-raw.sh )
echo "Submitted"
sleep 2

for i in ${!pheno_list_iter[@]}; do
    for m in ${n_snps_list[@]}; do
	pheno_file="${PHENO_DIR}/${pheno_list[$i]}_${snp_sets[$i]}_top${m}.raw"
	
	model_file="pkl/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_baseline.pkl"
	if [ ! -f $model_file ]; then
	    echo "Submitting ${model_file}..."
	    qsub -W depend=afterok:$pheno_job -v pheno_file=$pheno_file,model_file=$model_file,self_interact=False,anchor=False,n_restarts=10 fit_ukb.sh
	    sleep 2
	fi
	
	model_file="pkl/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_anchor_selfinteract.pkl"
	if [ ! -f $model_file ]; then
            echo "Submitting ${model_file}..."
            qsub -W depend=afterok:$pheno_job -v pheno_file=$pheno_file,model_file=$model_file,self_interact=True,anchor=True,n_restarts=10 fit_ukb.sh
            sleep 2
        fi
        
	model_file="pkl/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_anchor.pkl"
	if [ ! -f $model_file ]; then
            echo "Submitting ${model_file}..."
            qsub -W depend=afterok:$pheno_job -v pheno_file=$pheno_file,model_file=$model_file,self_interact=False,anchor=True,n_restarts=10 fit_ukb.sh
            sleep 2
        fi
    done
done
