import pandas as pd
import argparse
from sklearn.model_selection import KFold



def create_folds(dataset):
    print(f'creating folds for dataset: {dataset}')
    df = pd.read_csv(f'./{dataset}/geno/yeast.fam', sep = ' ', header=None)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    for train_idx, test_idx in kf.split(df[0]):
        train = df.iloc[train_idx, [0, 1]]
        test = df.iloc[test_idx, [0, 1]]
        train.to_csv(f'./{dataset}/folds/train{fold}.csv', index=False, header=None, sep='\t')
        test.to_csv(f'./{dataset}/folds/test{fold}.csv', index=False, header=None, sep='\t')
        fold += 1

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="bloom13", type=str)
    args = parser.parse_args()

    create_folds(args.data)


