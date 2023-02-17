import sys
sys.path.insert(0,'../model')

from datasets import *
import numpy as np
import pickle as pkl
import argparse
import csv

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("ktrue", help="number of latent pathways", type=int)
    parser.add_argument("repeat", help="index of repeat", type=int)
    parser.add_argument("seed", help="seed for rng", type=int)
    parser.add_argument("param", help="which param to benchmark", type=str)

    parser.add_argument('--self', default=True, action='store_true')
    parser.add_argument('--no-self', dest='self', action='store_false')

    args = parser.parse_args()

    k = args.ktrue
    repeat = args.repeat
    self = args.self
    N = 1000

    suffix = '_repeat%d' % (repeat)

    if self:
        prefix = './%s_k%d_self/' % (args.param, k)
    else:
        prefix = './%s_k%d/' % (args.param, k)

    oos = {}
    data = {}

    np.random.seed(args.seed)

    #if args.param in ['percent_uncoordinated', 'anchor_strength']:
    #    param_list = np.linspace(0, 0.9, 10).round(1)
    #else:
    #    param_list = np.linspace(0.1, 1, 10).round(1)
    
    if args.param == 'snp':
        param_list = np.array([5, 10, 20, 50, 100, 500])
    elif args.param == 'N':
        param_list = np.array([100, 500, 1000, 5000, 10000, 50000, 100000])
        param_list = np.array([100, 500, 1000, 5000, 10000])
    else:
        param_list = np.linspace(0, 1, 11).round(1)
    

    for param in param_list:
        p=1e-8 if param==0 else param
        if args.param == 'percent_uncoordinated':
            data[param] = SimDataset(N, 20, k=k, h2=0.5, noise_omega=p, self_interactions=self)
        elif args.param == 'heritability':
            data[param] = SimDataset(N, 20, k=k, h2=p, self_interactions=self)
        elif args.param == 'additive_var':
            data[param] = SimDataset(N, 20, k=k, h2=0.5, additive_model_var=p, self_interactions=self)
        elif args.param == 'dominance':
            data[param] = SimDataset(N, 20, k=k, h2=0.5, additive_model_var=p, noise_omega=1, dominance=True, self_interactions=self)
        elif args.param == 'anchor_strength':
            data[param] = SimDataset(N, 20, k=k, h2=0.5, anchor_strength=p, self_interactions=self)
        elif args.param == 'snp':
            data[param] = SimDataset(N, p, k=k, h2=0.5, self_interactions=self)
        elif args.param == 'N':
            data[param] = SimDataset(p, 20, k=k, h2=0.5, self_interactions=self)
        else: 
            print('poorly specified parameter')
            exit()

        oos[param] = generateOOS(data[param], N)
        
    with open(prefix + 'training' + suffix + '.pkl', 'wb') as f:
        pkl.dump(data, f)
        
    with open(prefix + 'validation' + suffix + '.pkl', 'wb') as f:
        pkl.dump(oos, f)
