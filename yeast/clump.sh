#!/bin/bash

#SBATCH --job-name=assoc
#SBATCH --output=job-output/assoc.%A.out
#SBATCH --error=job-output/assoc.%A.err
#SBATCH --time=00:10:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --account=pi-andyd

module load plink
pheno=$1
training=$2
prefix=$3

plink \
    --silent \
    --bfile ${prefix}/geno/yeast \
    --linear 'hide-covar' \
    --pheno ${prefix}/pheno/yeast.phen \
    --mpheno ${pheno} \
    --allow-no-sex \
    --keep $training \
    --out ${prefix}/sumstats/pheno${pheno}

plink \
    --silent \
    --bfile ${prefix}/geno/yeast \
    --clump-p1 0.05 \
    --clump-r2 0.2 \
    --clump-kb 250 \
    --clump ${prefix}/sumstats/pheno${pheno}.assoc.linear \
    --clump-snp-field SNP \
    --clump-field P \
    --keep $training \
    --out ${prefix}/clumped/pheno${pheno}

awk 'NR!=1{print $3}' ${prefix}/clumped/pheno${pheno}.clumped > \
    ${prefix}/clumped/pheno${pheno}.kept.snps

awk 'NR!=1{print $4}' ${prefix}/clumped/pheno${pheno}.clumped > \
    ${prefix}/clumped/pheno${pheno}.kept.snps.idx

plink \
    --silent \
    --export A \
    --extract ${prefix}/clumped/pheno${pheno}.kept.snps \
    --keep $training \
    --out ${prefix}/input/train${pheno} \
    --bfile ${prefix}/geno/yeast \
    --pheno ${prefix}/pheno/yeast.phen \
    --mpheno ${pheno}

plink \
    --silent \
    --export A \
    --extract ${prefix}/clumped/pheno${pheno}.kept.snps \
    --remove $training \
    --out ${prefix}/input/val${pheno} \
    --bfile ${prefix}/geno/yeast \
    --pheno ${prefix}/pheno/yeast.phen \
    --mpheno ${pheno}
