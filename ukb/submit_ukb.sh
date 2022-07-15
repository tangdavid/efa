#!/bin/bash

source /gpfs/data/ukb-share/dahl/jerome/my-base-env/bin/activate /gpfs/data/ukb-share/dahl/jerome/my-base-env/envs/epistasis

cd /gpfs/data/ukb-share/dahl/davidtang/epistasis-factorization/ukb

PHENO_DIR="/gpfs/data/ukb-share/dahl/jerome/extracted_phenotypes/epistasis_pheno"
# hba1c
# pheno_list="hba1c"
# pheno_ids="30750"
# pheno_baskets="29329"
# snp_sets="xue-t2d"

# urate
pheno_list="urate"
pheno_ids="30880-0.0"
pheno_baskets="29329"
snp_sets="sinarm-urate-range"

# igf-1
#pheno_list="igf1"
#pheno_ids="30770-0.0"
#pheno_baskets="29329"
#snp_sets="sinarm-igf1-range"

# male testosterone
#pheno_list="male_testosterone"
#pheno_ids="30850-0.0"
#pheno_baskets="29329"
#snp_sets="sinarm-maletest-range"

# female testosterone
#pheno_list="female_testosterone"
#pheno_ids="30850-0.0"
#pheno_baskets="29329"
#snp_sets="sinarm-femaletest-range"




#n_snps="6;10;20;100"
n_snps="100"
n_permutations=1000
split_seed=42
folds=2
restarts=20


n_snps_list=(${n_snps//;/ })
pheno_list_iter=(${pheno_list//;/ })
# run gen-plink-raw here, pass args -> use qsub -W depend=afterok:$pheno_job -v ...
# echo "Submitting pheno job..."
# pheno_job=$( qsub -v pheno_list=$pheno_list,pheno_ids=$pheno_ids,pheno_baskets=$pheno_baskets,snp_sets=$snp_sets,n_snps=$n_snps gen-plink-raw.sh )
# echo "Submitted"
# sleep 10

for i in ${!pheno_list_iter[@]}; do
    for m in ${n_snps_list[@]}; do
	pheno_file="${PHENO_DIR}/${pheno_list[$i]}_${snp_sets[$i]}_top${m}.raw"
	
	model_file="pkl/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_baseline.pkl"
	if [ ! -f $model_file ]; then
	    echo "Submitting ${model_file}..."
	    qsub -v pheno_file=$pheno_file,model_file=$model_file,self_interact=False,anchor=False,n_restarts=$restarts,seed=$split_seed,k=$folds,permute=False fit_ukb.sh
	    sleep 2
	fi

	for p in $(seq 0 $n_permutations); do
	    mkdir -p "pkl/permutations/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_baseline"
	    p_file="pkl/permutations/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_baseline/perm${p}.pkl" 
	    if [ ! -f $p_file ]; then
		echo "Submitting permutation..."
		qsub -v pheno_file=$pheno_file,model_file=$p_file,self_interact=False,anchor=False,n_restarts=$restarts,seed=$split_seed,k=$folds,permute=True fit_ukb.sh
		sleep 2
	    fi
	done
	
	# model_file="pkl/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_anchor_selfinteract.pkl"
	# if [ ! -f $model_file ]; then
        #     echo "Submitting ${model_file}..."
        #     qsub -v pheno_file=$pheno_file,model_file=$model_file,self_interact=True,anchor=True,n_restarts=$restarts,seed=$split_seed,k=$folds,permute=False fit_ukb.sh
        #     sleep 2
        # fi
        
	# model_file="pkl/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_anchor.pkl"
	# if [ ! -f $model_file ]; then
        #     echo "Submitting ${model_file}..."
        #     qsub -v pheno_file=$pheno_file,model_file=$model_file,self_interact=False,anchor=True,n_restarts=$restarts,seed=$split_seed,k=$folds,permute=False fit_ukb.sh
        #     sleep 2
        # fi
    done
done
