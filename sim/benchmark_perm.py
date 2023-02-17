import sys
sys.path.insert(0,'../model')

from models import UncoordinatedModel, CoordinatedModel, AdditiveModel
from tools import tools
from datasets import SimDataset
import pickle as pkl
import argparse
import csv


def benchmark(prefix, suffix):
    ce_list = list()
    am_list = list()
    ep_list = list()

    with open(prefix + 'training' + suffix + '.pkl', 'rb') as f:
        training = pkl.load(f)

    with open(prefix + 'validation' + suffix + '.pkl', 'rb') as f:
        validation = pkl.load(f)

    for data in [training, validation]:

        ce = CoordinatedModel(k = 2)
        am = AdditiveModel()
        ep = UncoordinatedModel()

        ce.fitModel(data, selfInteractions = False, anchors = False, restarts=20)
        am.fitModel(data)
        ep.fitModel(data)

        ce_list.append(ce)
        am_list.append(am)
        ep_list.append(ep)
            
    with open(prefix + 'models' + suffix + '.pkl', 'wb') as f:
        pkl.dump(ce_list, f)
        pkl.dump(am_list, f)
        pkl.dump(ep_list, f)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repeat", help="index of repeat", type=int)
    parser.add_argument("name", help="dominance or LD", type=str)

    parser.add_argument('--null', default=False, action='store_true')
    parser.add_argument('--no-null', dest='null', action='store_false')
    args = parser.parse_args()

    prefix = './%s_perm/' % (args.name)
    if args.null: prefix += 'null/'
    suffix = '_%d' % (args.repeat)
    benchmark(prefix, suffix) 


