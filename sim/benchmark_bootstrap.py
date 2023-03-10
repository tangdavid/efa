import sys
sys.path.insert(0,'../model')

import numpy as np
from datasets import SimDataset, RealDataset, SimDatasetLD
from models import AdditiveModel, CoordinatedModel
import argparse
import pickle as pkl
import os

def main(args):
    N = 10000
    M = 10
    if args.name == 'dominance':
        data = SimDataset(
            N, M, 
            k=2, 
            h2=0.1, 
            additive_model_var=1 - args.h2_ep, 
            noise_omega=1, 
            dominance=True
        )
        true_weights = 0
    elif args.name == 'efa':
        data = SimDataset(
            N, M, 
            k=2, 
            self_interactions=False,
            h2=0.1, 
            additive_model_var=1 - args.h2_ep
        )
        if args.h2_ep == 0:
            true_weights = 0
        else:
            true_weights = data.weights[0, 1]
    elif args.name == 'LD':
        data = SimDatasetLD(
            N, M, 
            r2=0.49, 
            h2=0.1
        )
        true_weights = 0

    fname = f'{args.name}_{int(args.h2_ep * 100)}'
    path = f'./pkl/{fname}'
    if not os.path.exists(path):
        os.makedirs(path)

    fname = f'{path}/repeat{args.repeat}.pkl'

    print('=' * 80)
    print(f'sim type: {args.name}')
    print(f'epistatic h2: {args.h2_ep}')
    print(f'repeat: {args.repeat}')
    print(f'out: {fname}')
    print('=' * 80, flush=True)
    

    data = RealDataset(geno = data.geno, pheno = data.pheno)
    add = AdditiveModel()
    add.fitModel(data)

    fit = lambda x, y: x.fitModel(
        y, 
        algo = 'coord', 
        progress = False, 
        restarts = 1, 
        init_noise = 0.1, 
        additive_init=True, 
        tol = 1e-9,
        min_iter = 1,
        max_iter = 5000
    )

    efa = CoordinatedModel(k = 2, sink = False)
    n_bs = 500
    res = np.zeros(n_bs)
    res_omega = np.zeros(n_bs)
    conv = np.zeros(n_bs, dtype=bool)
    fit(efa, data)
    full_fit = efa.weights[0, 1]

    for i in range(n_bs):
        if (i + 1) % 100 == 0: print(f'bs sample {i+1}...', flush=True)
        data_bs = data.bootstrap()
        efa = CoordinatedModel(k = 2, sink = False)
        fit(efa, data_bs)
        omega_signed = np.linalg.norm(efa.pathways[:, 0] * efa.pathways[:, 1])

        res[i] = efa.weights[0, 1]
        res_omega[i] = efa.weights[0, 1] * omega_signed
        conv[i] = efa.conv

    lower = np.quantile(res[conv], 0.025)
    median = np.quantile(res[conv], 0.5)
    upper = np.quantile(res[conv], 0.975)
    contained = (lower <= true_weights) and (upper >= true_weights)
    with open(fname, 'wb') as f:
        pkl.dump(contained, f)
        pkl.dump(median, f)
        pkl.dump(full_fit, f)
        pkl.dump(true_weights, f)
        pkl.dump(res, f)
        pkl.dump(conv, f)
        pkl.dump(res_omega, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repeat", help="index of repeat", type=int)
    parser.add_argument("name", help="additive, power, dominance or LD", type=str)
    parser.add_argument("h2_ep", help="epistatic variance", type=float, default=0)

    args = parser.parse_args()
    main(args) 
