import sys
sys.path.insert(0, './model/')


import pandas as pd
from datasets import RealDataset
from datasets import splitTrain, splitKFold
from models import CoordinatedModel, AdditiveModel, UncoordinatedModel
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pickle as pkl
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

    parser.add_argument('--sink', default=False, action='store_true')
    parser.add_argument('--no-sink', dest='sink', action='store_false')

    parser.add_argument("--algo", default="coord", type=str)

    args = parser.parse_args()

    PHENO_FILE=args.pheno
    MODEL_FILE=args.out
    IS_SELF_INTERACT=args.self
    IS_ANCHOR=args.anchor
    IS_PERMUTE=args.permute
    N_RESTARTS=args.restarts
    SINK=args.sink
    ALGO=args.algo
    K=args.k

    print("="*80)
    print("phenotype file: %s" %PHENO_FILE)
    print("output file: %s" %MODEL_FILE)
    print("number of pathways: %d" %K)
    print("number of restarts: %d" %N_RESTARTS)
    print("self interactions: %s" %str(IS_SELF_INTERACT))
    print("anchors: %s" %str(IS_ANCHOR))
    print("permuted data: %s" %str(IS_PERMUTE))
    print("="*80)
    sys.stdout.flush()

    start = time.time()
    print('Loading Data...', flush=True)
    data = RealDataset(infile = PHENO_FILE)
    if IS_PERMUTE: data.permute()

    print('Fitting Model...',flush=True)

    efa = CoordinatedModel(k = K)
    efa.fitModel(
        data, 
        algo=ALGO,
        restarts=N_RESTARTS, 
        additive_init=True,
        self_interactions=IS_SELF_INTERACT, 
        anchors=IS_ANCHOR,
        sink=SINK
    )
    add = AdditiveModel()
    add.fitModel(data)

    with open(MODEL_FILE, 'wb') as f:
        pkl.dump(efa, f)
        pkl.dump(add, f)

    end = time.time()
    print('Done!')
    print('Time: %f' % (end - start))
