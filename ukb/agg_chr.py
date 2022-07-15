import sys
import pandas as pd
from os.path import exists

DIR = "/gpfs/data/ukb-share/dahl/jerome/extracted_phenotypes/epistasis_pheno/"
PREFIX = sys.argv[1]

print(PREFIX)

# combine PREFIX_chr{i}.raw -> PREFIX.raw

# TODO: make df

df = pd.DataFrame()
for i in range(1,22+1):
    if (exists(DIR+PREFIX+f"_chr{i}.raw")):
        chr_df = pd.read_csv(DIR+PREFIX+f"_chr{i}.raw", header=0, sep='\t', dtype=str)
        if df.empty:
            df = chr_df
        else:
            df = pd.merge(df, chr_df, on=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'])


# set anchor snps to last 2 columns
pheno = PREFIX[0:PREFIX.find('_')]
if pheno == 'hba1c':
    a1 = "rs7903146_T" # TCF7L2
    a2 = "rs7185735_G" # FTO
    df.insert(len(df.columns)-1, a1, df.pop(a1))
    df.insert(len(df.columns)-1, a2, df.pop(a2))
    
df.to_csv(DIR+PREFIX+".raw", sep='\t', index=False)
