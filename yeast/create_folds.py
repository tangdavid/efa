import pandas as pd
from sklearn.model_selection import KFold

dataset = 'bloom13'
df = pd.read_csv(f'./{dataset}/geno/yeast.fam', sep = ' ', header=None)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold = 0
for train_idx, test_idx in kf.split(df[0]):
    train = df.iloc[train_idx, 0]
    test = df.iloc[test_idx, 0]
    train.to_csv(f'./{dataset}/folds/train{fold}.csv', index=False, header=None)
    test.to_csv(f'./{dataset}/folds/test{fold}.csv', index=False, header=None)
    fold += 1
