from sim import *
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument("k", help="number of latent pathways", type=int)
parser.add_argument("repeat", help="index of repeat", type=int)
parser.add_argument("seed", help="seed for rng", type=int)
args = parser.parse_args()

res = list()
k = args.k
repeat = args.repeat

suffix = '_k%d_repeat%d' % (k, repeat)
prefix = './ce/'

oos = {}
data = {}

np.random.seed(args.seed)

for param in np.linspace(0, 0.9, 10).round(1):
    data[param] = Dataset(10000, 20, k=k, h2=0.7, noise_omega=param)
    oos[param] = tools.generateOOS(data[param], 10000)
    
with open(prefix + 'training' + suffix + '.pkl', 'wb') as f:
    pkl.dump(data, f)
    
with open(prefix + 'validation' + suffix + '.pkl', 'wb') as f:
    pkl.dump(oos, f)
