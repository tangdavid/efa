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

def main(args):

    PHENO_FILE=args.pheno
    MODEL_FILE=args.out
    IS_SELF_INTERACT=args.self
    IS_ANCHOR=args.anchor
    IS_PERMUTE=args.permute
    RINT=args.rint
    N_RESTARTS=args.restarts
    INIT_NOISE=args.init_noise
    SINK=args.sink
    ALGO=args.algo
    K=args.k
    BOOTSTRAP=args.bootstrap
    PERMUTE_SEED=args.permute_seed

    print("="*80)
    print("phenotype file: %s" %PHENO_FILE)
    print("output file: %s" %MODEL_FILE)
    print("number of pathways: %d" %K)
    print("number of restarts: %d" %N_RESTARTS)
    print("init noise: %f" %INIT_NOISE)
    print("self interactions: %s" %str(IS_SELF_INTERACT))
    print("anchors: %s" %str(IS_ANCHOR))
    print("permuted data: %s" %str(IS_PERMUTE))
    print("permuted seed: %s" %str(PERMUTE_SEED))
    print("rint: %s" %str(RINT))
    print("sink: %s" %str(SINK))
    print("="*80)
    sys.stdout.flush()

    start = time.time()


    print('Loading Data...', flush=True)
    data = RealDataset(infile = PHENO_FILE, rint = RINT)

    add = AdditiveModel()
    add.fitModel(data)
    if IS_PERMUTE: 
        print('Using Permuted Data...', flush=True)
        add_effects = add.predictPheno(data)
        data.permute(residualize=add_effects, seed=PERMUTE_SEED)

    if BOOTSTRAP: 
        print('Using Bootstrap Sample...', flush=True)
        data = data.bootstrap()

    print('Fitting Model...',flush=True)
    add.fitModel(data)
    efa = CoordinatedModel(k=K, sink=SINK)
    efa.fitModel(
        data, 
        algo=ALGO,
        restarts=N_RESTARTS, 
        init_noise=INIT_NOISE, 
        additive_init=True,
        progress=True,
        self_interactions=IS_SELF_INTERACT, 
        anchors=IS_ANCHOR,
        tol=1e-7,
        min_iter = 1,
        max_iter = 1000
    )

    with open(MODEL_FILE, 'wb') as f:
        pkl.dump(efa, f)
        pkl.dump(add, f)

    end = time.time()
    print('Done!')
    print('Time: %f' % (end - start))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pheno", help="phenotype file", type=str)
    parser.add_argument("--out", default="./ce_out.pkl", type=str)

    parser.add_argument('--k', default=2, type=int)
    parser.add_argument('--restarts', default=20, type=int)
    parser.add_argument('--init-noise', default=0.1, type=float)
    parser.add_argument('--permute-seed', type=int)

    parser.add_argument('--self', default=False, action='store_true')
    parser.add_argument('--no-self', dest='self', action='store_false')

    parser.add_argument('--anchor', default=False, action='store_true')
    parser.add_argument('--no-anchor', dest='anchor', action='store_false')

    parser.add_argument('--permute', default=False, action='store_true')
    parser.add_argument('--no-permute', dest='permute', action='store_false')

    parser.add_argument('--rint', default=False, action='store_true')
    parser.add_argument('--no-rint', dest='rint', action='store_false')

    parser.add_argument('--sink', default=False, action='store_true')
    parser.add_argument('--no-sink', dest='sink', action='store_false')

    parser.add_argument('--bootstrap', default=False, action='store_true')
    parser.add_argument('--no-bootstrap', dest='bootstrap', action='store_false')

    parser.add_argument("--algo", default="coord", type=str)

    args = parser.parse_args()
    main(args)

