import sys
sys.path.insert(0,'../model')

from datasets import *
from models import AdditiveModel
import numpy as np
import pickle as pkl
import argparse
import csv

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("repeat", help="index of repeat", type=int)
    parser.add_argument("seed", help="seed for rng", type=int)
    parser.add_argument("name", help="dominance or LD sim", type=str)

    parser.add_argument('--null', default=False, action='store_true')
    parser.add_argument('--no-null', dest='null', action='store_false')

    args = parser.parse_args()

    repeat = args.repeat
    N = 100000
    M = 100


    prefix = './%s_perm/' % (args.name)

    if args.name == 'dominance':
        data = SimDataset(N, M, k=2, h2=0.1, additive_model_var=0.99, noise_omega=1, dominance=True)
    elif args.name == 'additive':
        data = SimDataset(N, M, h2=0.1, additive_model_var=1)
    elif args.name == 'power':
        data = SimDataset(N, M, k=2, h2=0.1, additive_model_var=0.7)
    elif args.name == 'baseline':
        data = SimDataset(N, M, k=2, h2=1e-6, additive_model_var=1)
    elif args.name == 'LD':
        data = SimDatasetLD(N, M, r2=0.49, h2=0.1)

    oos = generateOOS(data, N)
    delattr(data, 'inter')
    delattr(oos, 'inter')

    full = concatDatasets(data, oos)
    add_model = AdditiveModel()
    add_model.fitModel(full)
    add_effect = add_model.predictPheno(full)

    if args.null:
        for i in range(1, 1000 + 1):
            print(f'permutation {i}...', flush=True)
            data.permute(residualize=add_effect[:N,:])
            oos.permute(residualize=add_effect[N:,:])
            suffix = '_%d' % (i)
            with open(prefix + 'null/training' + suffix + '.pkl', 'wb') as f:
                pkl.dump(data, f)
            with open(prefix + 'null/validation' + suffix + '.pkl', 'wb') as f:
                pkl.dump(oos, f)

    else:
        suffix = '_%d' % (repeat)
        with open(prefix + 'training' + suffix + '.pkl', 'wb') as f:
            pkl.dump(data, f)
        with open(prefix + 'validation' + suffix + '.pkl', 'wb') as f:
            pkl.dump(oos, f)
