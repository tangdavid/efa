import sys
sys.path.insert(0, './model/')


import pandas as pd
from tools import *
from datasets import RealDataset
from datasets import splitTrain, splitKFold
from models import CoordinatedModel, AdditiveModel, UncoordinatedModel
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import time
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pheno", help="phenotype file", type=str)
    parser.add_argument("--out", default="./ce_out.pkl", type=str)

    parser.add_argument('--k', default=2, type=int)
    parser.add_argument('--restarts', default=20, type=int)

    parser.add_argument('--self', default=False, action='store_true')
    parser.add_argument('--no-self', dest='self', action='store_false')

    parser.add_argument('--anchor', default=False, action='store_true')
    parser.add_argument('--no-anchor', dest='anchor', action='store_false')

    parser.add_argument('--permute', default=False, action='store_true')
    parser.add_argument('--no-permute', dest='permute', action='store_false')


    args = parser.parse_args()

    PHENO_FILE=args.pheno
    MODEL_FILE=args.out
    IS_SELF_INTERACT=args.self
    IS_ANCHOR=args.anchor
    N_RESTARTS=args.restarts
    PERMUTE=args.permute
    K=args.k

    print("="*80)
    print("phenotype file: %s" %PHENO_FILE)
    print("output file: %s" %MODEL_FILE)
    print("number of pathways: %d" %K)
    print("number of restarts: %d" %N_RESTARTS)
    print("self interactions: %s" %str(IS_SELF_INTERACT))
    print("anchors: %s" %str(IS_ANCHOR))
    if PERMUTE: print("using permuted phenotypes")
    print("="*80)
    exit()

    start = time.time()
    print('Loading Data...')
    data = RealDataset(infile = PHENO_FILE)

    print('Fitting Model...')

    ce = CoordinatedModel(k = 2)
    ce.fitModel(train, restarts=N_RESTARTS, selfInteractions=IS_SELF_INTERACT, anchors=IS_ANCHOR)

    with open(MODEL_FILE, 'wb') as f:
        pkl.dump(ce, f)

    end = time.time()
    print('Done!')
    print('Time: %f' % (end - start))
