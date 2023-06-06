#!/bin/bash

#SBATCH --job-name=assoc
#SBATCH --output=job-output/assoc.%A.out
#SBATCH --error=job-output/assoc.%A.err
#SBATCH --time=00:10:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --account=pi-andyd

module load plink
prefix=bloom15

for pheno in {1..20}
do
    plink \
        --bfile ${prefix}/geno/yeast \
        --linear 'hide-covar' \
        --pheno ${prefix}/pheno/yeast.phen \
        --mpheno ${pheno} \
        --allow-no-sex \
        --out ${prefix}/sumstats/pheno${pheno}

    plink \
        --bfile ${prefix}/geno/yeast \
        --clump-p1 0.05 \
        --clump-r2 0.2 \
        --clump-kb 250 \
        --clump ${prefix}/sumstats/pheno${pheno}.assoc.linear \
        --clump-snp-field SNP \
        --clump-field P \
        --out ${prefix}/clumped/pheno${pheno}

    awk 'NR!=1{print $3}' ${prefix}/clumped/pheno${pheno}.clumped > \
        ${prefix}/clumped/pheno${pheno}.kept.snps

    plink \
        --export A \
        --extract ${prefix}/clumped/pheno${pheno}.kept.snps \
        --out ${prefix}/input/pheno${pheno} \
        --bfile ${prefix}/geno/yeast \
        --pheno ${prefix}/pheno/yeast.phen \
        --mpheno ${pheno}

    plink \
        --export A \
        --out ${prefix}/raw/pheno${pheno} \
        --bfile ${prefix}/geno/yeast \
        --pheno ${prefix}/pheno/yeast.phen \
        --mpheno ${pheno}
done
echo "done!"
