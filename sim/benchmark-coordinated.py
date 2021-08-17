from sim import *
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument("k", help="number of latent pathways", type=int)
parser.add_argument("repeat", help="index of repeat", type=int)
parser.add_argument("param", help="index of repeat", type=str)
args = parser.parse_args()

res_CE = list()
res_additive = list()
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
    min_loss = float('inf')
    pathways = None

    for restart in range(20):
        coordinatedModel = CoordinatedModel()
        coordinatedModel.fitModel(data)
        loss = coordinatedModel.loss[-1]
        if loss < min_loss:
            min_loss = loss
            pathways = coordinatedModel.pathways           
            accPathways = coordinatedModel.evalPathwayAcc(data)
            accPheno = coordinatedModel.evalPhenoAcc(oos)
            accBeta = coordinatedModel.evalBetaAcc(data)
            accOmega = coordinatedModel.evalOmegaAcc(data)

    res.append([param,
                accPathways[0], 
                accPathways[1], 
                accPheno,
                accBeta,
                accOmega,
                k,
                repeat])

    print("done with param %0.1f" %param)

with open(prefix + 'coordinated' + suffix + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res: writer.writerow(row)
        
