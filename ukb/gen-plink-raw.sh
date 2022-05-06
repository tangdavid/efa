#!/bin/bash

#PBS -N epistasis_pheno
#PBS -S /bin/bash
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=4gb
#PBS -o $HOME/epistasis_pheno$PBS_JOBID.out
#PBS -e $HOME/epistasis_pheno$PBS_JOBID.err

source /gpfs/data/ukb-share/dahl/jerome/my-base-env/bin/activate /gpfs/data/ukb-share/dahl/jerome/my-base-env/envs/epistasis


# params
rows=50000

# make sure these have same index -- could replace with associative array
# pheno_list=("hba1c")
# pheno_ids=(30750)
# pheno_baskets=(29329)
# snp_sets=("xue-t2d")
pheno_list=(${pheno_list//;/ })
pheno_ids=(${pheno_ids//;/ })
pheno_baskets=(${pheno_baskets//;/ })
snp_sets=(${snp_sets//;/ })

PC_num=10 # 0,10,20,30,40
covariates="sab"
#n_snps=(5 10 20 100)
n_snps=(${n_snps//;/ })

# paths
PHENO_DIR="/gpfs/data/ukb-share/dahl/jerome/extracted_phenotypes"
COVAR_DIR="${PHENO_DIR}/covariates_${covariates}40PC"
GENO_DIR="/scratch/t.cri.jfreudenberg/genotypes/v3"
SNP_DIR="/gpfs/data/ukb-share/dahl/jerome/snp_lists"

# scripts
PHENO_EXTRACT_SCRIPT="/gpfs/data/ukb-share/dahl/jerome/uchicago-ukb/extract_pheno.py"
PHENO_RESID_SCRIPT="/gpfs/data/ukb-share/dahl/jerome/epistasis-factorization/ukb/resid.R"
CHR_AGG_SCRIPT="/gpfs/data/ukb-share/dahl/jerome/epistasis-factorization/ukb/agg_chr.py"

# data
#SAMPLES_FILE="${PHENO_DIR}/whitebritish_ids.txt"
SAMPLES_FILE="${PHENO_DIR}/10000sample0.txt"

for i in ${!pheno_list[@]}; do
    echo "Running on ${pheno_list[$i]}..."
    
    for m in ${n_snps[@]}; do
	# extract top m snps from list
	if [ ! -f "${SNP_DIR}/${snp_sets[$i]}_top${m}.txt" ]; then
	    echo "Making top ${m} snp list..."
	    head -n $m "${SNP_DIR}/${snp_sets[$i]}.txt" > "${SNP_DIR}/${snp_sets[$i]}_top${m}.txt"
	fi
	snp_file="${SNP_DIR}/${snp_sets[$i]}_top${m}.txt"

	# get phenotype file
	pheno_file="${PHENO_DIR}/${pheno_list[$i]}/${pheno_list[$i]}${pheno_baskets[$i]}.pheno"
	if [ ! -f $pheno_file ]; then
	    echo "Extracting phenotype..."
	    python3 $PHENO_EXTRACT_SCRIPT ${pheno_ids[$i]} -n ${pheno_list[$i]} -c -r $rows -t $PHENO_DIR
	fi

	# project covariates
	covar_file="${COVAR_DIR}/covariates_${covariates}${PC_num}PC27702.pheno"
	if [ ! -f "${COVAR_DIR}/${pheno_list[$i]}${PC_num}.pheno" ]; then
	    echo "Residualizing phenotype..."
	    Rscript $PHENO_RESID_SCRIPT $pheno_file $covar_file "${COVAR_DIR}/${pheno_list[$i]}${PC_num}.pheno"
	    echo "Wrote to ${COVAR_DIR}/${pheno_list[$i]}${PC_num}.pheno"
	fi

	# gen plink raw by chr if not exist
	if [ ! -f "${PHENO_DIR}/epistasis_pheno/${pheno_list[$i]}_${snp_sets[$i]}_top${m}.raw" ]; then
	    for c in $(seq 1 22); do
		if [ ! -f "${PHENO_DIR}/epistasis_pheno/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_chr${c}.raw" ]; then  
		    echo "Extracting chr ${c}"
		    
		    /gpfs/data/ukb-share/plink2 \
	    		--pfile ${GENO_DIR}/ukb_imp_chr${c}_v3 \
	    		--extract ${snp_file} \
	    		--out ${PHENO_DIR}/epistasis_pheno/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_chr${c} \
	    		--pheno ${COVAR_DIR}/${pheno_list[$i]}${PC_num}.pheno \
			--export A \
			--prune \
			--keep $SAMPLES_FILE
		    
		fi
	    done

	    # aggregate chr
	    echo "Combining chr..."
	    python $CHR_AGG_SCRIPT ${pheno_list[$i]}_${snp_sets[$i]}_top${m}
	    rm ${PHENO_DIR}/epistasis_pheno/${pheno_list[$i]}_${snp_sets[$i]}_top${m}_chr*.raw
	    mv ${PHENO_DIR}/epistasis_pheno/*.log ${PHENO_DIR}/epistasis_pheno/logs
	fi
    done
done

