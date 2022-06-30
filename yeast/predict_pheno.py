import sys
sys.path.insert(0,'../model')

from models import UncoordinatedModel, CoordinatedModel, AdditiveModel
import numpy as np
from tools import tools
from datasets import RealDataset
from datasets import splitTrain, splitKFold
import pickle as pkl
import argparse
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pheno", help="phenotype number", type=int)
    parser.add_argument("dataset", help="which dataset", type=str)
    args = parser.parse_args()

    print("datset: %s" %args.dataset)
    print("pheno: %d" %args.pheno)
    sys.stdout.flush()

    data = RealDataset(infile = '%s/input/pheno%d.raw' % (args.dataset, args.pheno), delim=' ')
    folds = splitKFold(data, folds=10, seed=42)
    
    ide_acc = list()
    ce2_acc = list()
    ce3_acc = list()
    ce4_acc = list()
    adr_acc = list()
    adf_acc = list()
    epf_acc = list()
    epr_acc = list()
    adr_raw_acc = list()
    epr_raw_acc = list()

    i = 0
    print('clumped data')
    sys.stdout.flush()
    for train, test in folds:
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

        ide_acc.append(ide.evalPhenoAcc(test))
        ce2_acc.append(ce2.evalPhenoAcc(test))
        ce3_acc.append(ce3.evalPhenoAcc(test))
        ce4_acc.append(ce4.evalPhenoAcc(test))
        adf_acc.append(adf.evalPhenoAcc(test))
        adr_acc.append(adr.evalPhenoAcc(test))
        epf_acc.append(epf.evalPhenoAcc(test))
        epr_acc.append(epr.evalPhenoAcc(test))

        i += 1
        print("done with fold %d" %i)
        sys.stdout.flush()

    print('raw data')
    sys.stdout.flush()

    data = RealDataset(infile = '%s/raw/pheno%d.raw' % (args.dataset, args.pheno), delim=' ')
    folds = splitKFold(data, folds=10, seed=42)
    for train, test in folds:
        adr = AdditiveModel()
        epr = UncoordinatedModel()

        adr.fitMLE(train)
        epr.fitMLE(train)

        adr_raw_acc.append(adr.evalPhenoAcc(test))
        epr_raw_acc.append(epr.evalPhenoAcc(test))

        i += 1
        print("done with fold %d" %i)
        sys.stdout.flush()
   
    res = (args.pheno, 
           np.mean(ide_acc), np.std(ide_acc)/np.sqrt(10),
           np.mean(ce2_acc), np.std(ce2_acc)/np.sqrt(10),
           np.mean(ce3_acc), np.std(ce3_acc)/np.sqrt(10),
           np.mean(ce4_acc), np.std(ce4_acc)/np.sqrt(10),
           np.mean(adf_acc), np.std(adf_acc)/np.sqrt(10),
           np.mean(adr_acc), np.std(adr_acc)/np.sqrt(10),
           np.mean(epf_acc), np.std(epf_acc)/np.sqrt(10),
           np.mean(epr_acc), np.std(epr_acc)/np.sqrt(10),
           np.mean(adr_raw_acc), np.std(adr_raw_acc)/np.sqrt(10),
           np.mean(epr_raw_acc), np.std(epr_raw_acc)/np.sqrt(10))

    with open('%s/output/pheno%d.csv' % (args.dataset, args.pheno), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(res)

if __name__=='__main__':
    main()
