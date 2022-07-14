import sys
sys.path.insert(0,'../model')

from models import UncoordinatedModel, CoordinatedModel, AdditiveModel
import numpy as np
import pandas as pd
from datasets import RealDataset
from datasets import splitKFold
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pheno", help="phenotype number", type=int)
    parser.add_argument("dataset", help="which dataset", type=str)
    args = parser.parse_args()

    dataset = args.dataset
    pheno = args.pheno

    print("dataset: %s" %args.dataset)
    print("pheno: %d" %args.pheno)
    sys.stdout.flush()
    
    res = np.zeros((10, 12))
    cols = ['pheno',
            'ide', 'ce2', 'ce3', 'ce4',
            'adr', 'adf', 'epf', 'epr',
            'adr_raw', 'epr_raw',
            'fold']

    print('clumped data')
    sys.stdout.flush()
    for fold in range(10):
        train_file = f'{dataset}/folds/train{fold}.csv'
        subprocess.call(['bash', './clump.sh', str(pheno), train_file, dataset])
        train = RealDataset(infile = f'{dataset}/input/train{pheno}.raw', delim=' ')
        test = RealDataset(infile = f'{dataset}/input/val{pheno}.raw', delim=' ')
        
        ide = CoordinatedModel(k = 2)
        ce2 = CoordinatedModel(k = 2)
        ce3 = CoordinatedModel(k = 3)
        ce4 = CoordinatedModel(k = 4)
        adf = AdditiveModel()
        adr = AdditiveModel()
        epf = UncoordinatedModel()
        epr = UncoordinatedModel()

        ide.fitModel(train, restarts=20, selfInteractions=False, anchors=False)
        ce2.fitModel(train, restarts=20, selfInteractions=True, anchors=False)
        ce3.fitModel(train, restarts=20, selfInteractions=True, anchors=False)
        ce4.fitModel(train, restarts=20, selfInteractions=True, anchors=False)
        adf.fitModel(train)
        adr.fitModel(train, random_effects=True)
        epf.fitModel(train)
        epr.fitModel(train, random_effects=True)

        res[fold][0] = pheno
        res[fold][1] = ide.evalPhenoAcc(test)
        res[fold][2] = ce2.evalPhenoAcc(test)
        res[fold][3] = ce3.evalPhenoAcc(test)
        res[fold][4] = ce4.evalPhenoAcc(test)
        res[fold][5] = adf.evalPhenoAcc(test)
        res[fold][6] = adr.evalPhenoAcc(test)
        res[fold][7] = epf.evalPhenoAcc(test)
        res[fold][8] = epr.evalPhenoAcc(test)
        
        print("done with fold %d" % fold)
        sys.stdout.flush()

    print('raw data')
    sys.stdout.flush()

    data = RealDataset(infile = '%s/raw/pheno%d.raw' % (args.dataset, args.pheno), delim=' ')
    folds = splitKFold(data, folds=10, seed=42)
    fold = 0
    for train, test in folds:
        adr = AdditiveModel()
        epr = UncoordinatedModel()

        adr.fitMLE(train)
        epr.fitMLE(train)

        res[fold][9] =adr.evalPhenoAcc(test)
        res[fold][10] =epr.evalPhenoAcc(test)
        res[fold][11] = fold

        fold += 1
        print("done with fold %d" %fold)
        sys.stdout.flush()

    outfile = '%s/output/pheno%d.csv' % (dataset, pheno)
    pd.DataFrame(data=res, columns=cols).to_csv(outfile, index=None)

if __name__=='__main__':
    main()
