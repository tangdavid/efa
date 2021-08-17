from sim import *
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument("k", help="number of latent pathways", type=int)
parser.add_argument("repeat", help="index of repeat", type=int)
parser.add_argument("param", help="which param to benchmark", type=str)
args = parser.parse_args()

res = list()
k = args.k
repeat = args.repeat
prefix = './%s/' % args.param
suffix = '_k%d_repeat%d' % (k, repeat)

with open(prefix + 'training' + suffix + '.pkl', 'rb') as f:
    training = pkl.load(f)

with open(prefix + 'validation' + suffix + '.pkl', 'rb') as f:
    validation = pkl.load(f)

for param in training.keys():
    data = training[param]
    oos = validation[param]
            
    uncoordinatedModel = UncoordinatedModel()
    uncoordinatedModel.fitModel(data)
    accPheno = uncoordinatedModel.evalPhenoAcc(oos)
    accBeta = uncoordinatedModel.evalBetaAcc(data)
    accOmega = uncoordinatedModel.evalOmegaAcc(data)
    
    res.append([param, accPheno, accBeta, accOmega, k, repeat])
    print("done with param %0.1f" %param)
        
with open(prefix + 'uncoordinated' + suffix + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res: writer.writerow(row)
        