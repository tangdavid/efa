import sys
sys.path.insert(0,'../model')

from models import *
from tools import *
from datasets import *
import argparse
import csv

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("ktrue", help="number of latent pathways", type=int)
    parser.add_argument("repeat", help="index of repeat", type=int)
    parser.add_argument("seed", help="seed for rng", type=int)
    parser.add_argument("param", help="which param to benchmark", type=str)
    args = parser.parse_args()

    k = args.ktrue
    repeat = args.repeat

    suffix = '_repeat%d' % (repeat)
    prefix = './%s_k%d/' % (args.param, k)

    oos = {}
    data = {}

    np.random.seed(args.seed)

    if args.param in ['percent_uncoordinated', 'anchor_strength']:
        param_list = np.linspace(0, 0.9, 10).round(1)
    else:
        param_list = np.linspace(0.1, 1, 10).round(1)

    for param in param_list:
        if args.param == 'percent_uncoordinated':
            data[param] = SimDataset(10000, 20, k=k, noise_omega=param)
        elif args.param == 'heritability':
            data[param] = SimDataset(10000, 20, k=k, h2=param)
        elif args.param == 'additive_var':
            data[param] = SimDataset(10000, 20, k=k, additive_model_var=param)
        elif args.param == 'anchor_strength':
            data[param] = SimDataset(10000, 20, k=k, anchor_strength=param)
        else: 
            print('poorly specified parameter')
            exit()

        oos[param] = generateOOS(data[param], 10000)
        
    with open(prefix + 'training' + suffix + '.pkl', 'wb') as f:
        pkl.dump(data, f)
        
    with open(prefix + 'validation' + suffix + '.pkl', 'wb') as f:
        pkl.dump(oos, f)
