import sys
sys.path.insert(0, '../model/')


import pandas as pd
from tools import *
from datasets import RealDataset
from datasets import splitTrain, splitKFold
from models import CoordinatedModel, AdditiveModel, UncoordinatedModel
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import time

PHENO_FILE=sys.argv[1]
MODEL_FILE=sys.argv[2]
IS_SELF_INTERACT=sys.argv[3]=="True"
IS_ANCHOR=sys.argv[4]=="True"
N_RESTARTS=int(sys.argv[5])
SPLIT_SEED=int(sys.argv[6])
PERMUTE=sys.argv[7]=="True"

start = time.time()
print('Loading Data...')

data = RealDataset(infile = PHENO_FILE)

if PERMUTE:
    print('Using Permuted Data...')
    data.permute()

folds = splitKFold(data, folds=2, seed=SPLIT_SEED)

ce_list=list()

print('Fitting Model...')
for split,_ in folds:
    ce = CoordinatedModel(k = 2)
    ce.fitModel(split, restarts=N_RESTARTS, selfInteractions=IS_SELF_INTERACT, anchors=IS_ANCHOR)
    am.fitModel(split)
    ep.fitModel(split)

    ce_list.append(ce)
    am_list.append(am)
    ep_list.append(ep)

R2=stats.pearsonr(ce[0].omega.reshape(-1,), ce[1].omega.reshape(-1,))[0] ** 2

with open(MODEL_FILE, 'wb') as f:
    pkl.dump(R2, f)
    pkl.dump(ce_list, f)
    pkl.dump(am_list, f)
    pkl.dump(ep_list, f)

end = time.time()
print('Done!')
print('Time: %f' % (end - start))
