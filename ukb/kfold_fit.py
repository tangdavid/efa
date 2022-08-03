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
FOLDS=int(sys.argv[7])
PERMUTE=sys.argv[8]=="True"

start = time.time()
print('Loading Data...')
data = RealDataset(infile = PHENO_FILE)

folds = splitKFold(data, folds=FOLDS, seed=SPLIT_SEED)

ce_list=list()
am_list=list()
ep_list=list()
ce_acc =list()
am_acc = list()
ep_acc = list()

print('Fitting Model...')
for train, test in folds:
    print('Next fold...')

    if PERMUTE:
        print('Using Permuted Data...')
        train.permute()

    ce = CoordinatedModel(k = 2)
    am = AdditiveModel()
    ep = UncoordinatedModel()
    
    ce.fitModel(train, restarts=N_RESTARTS, selfInteractions=IS_SELF_INTERACT, anchors=IS_ANCHOR)
    am.fitModel(train)
    ep.fitModel(train)

    ce_list.append(ce)
    am_list.append(am)
    ep_list.append(ep)
    ce_acc.append(ce.evalPhenoAcc(test))
    am_acc.append(am.evalPhenoAcc(test))
    ep_acc.append(ep.evalPhenoAcc(test))

res2_mean = (np.mean(ce_acc), np.mean(am_acc), np.mean(ep_acc))
res2_std = (np.std(ce_acc), np.std(am_acc), np.std(ep_acc))

with open(MODEL_FILE, 'wb') as f:
    pkl.dump(ce_list, f)
    pkl.dump(am_list, f)
    pkl.dump(ep_list, f)
    pkl.dump(ce_acc, f)
    pkl.dump(am_acc, f)
    pkl.dump(ep_acc, f)
    pkl.dump(res2_mean, f)
    pkl.dump(res2_std, f)

end = time.time()
print('Done!')
print('Time: %f' % (end - start))