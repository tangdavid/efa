import sys
import pandas as pd
from os.path import exists

DIR = "/gpfs/data/ukb-share/dahl/jerome/extracted_phenotypes/epistasis_pheno/"
PREFIX = sys.argv[1]

print(PREFIX)

# combine PREFIX_chr{i}.raw -> PREFIX.raw

# TODO: make df

df = pd.DataFrame()
print(df)
for i in range(2,22+1):
    if (exists(DIR+PREFIX+f"_chr{i}.raw")):
        chr_df = pd.read_csv(DIR+PREFIX+f"_chr{i}.raw", header=0, sep='\t', dtype=str)
        if df.empty:
            df = chr_df
        else:
            df = pd.merge(df, chr_df, on=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'])
        print(df)
            
df.to_csv(DIR+PREFIX+".raw", sep='\t', index=False)
