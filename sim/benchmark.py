from sim import *
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument("k", help="number of latent pathways", type=int)
parser.add_argument("repeat", help="index of repeat", type=int)
args = parser.parse_args()

res_CE = list()
res_additive = list()
k = args.k
repeat = args.repeat
prefix = './h2/'
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
        coordinatedModel.gradDescent(data)
        loss = coordinatedModel.loss[-1]
        if loss < min_loss:
            min_loss = loss
            pathways = coordinatedModel.pathways           
            accPathwaysCE = coordinatedModel.evalPathwayAcc(data)
            accPhenoCE = coordinatedModel.evalPhenoAcc(data)
            accBetaCE = coordinatedModel.evalBetaAcc(data)
            
    additiveModel = AdditiveModel()
    additiveModel.fitLinearRegression(data)
    accPhenoAdditive = additiveModel.evalPhenoAcc(data)
    accBetaAdditive = additiveModel.evalBetaAcc(data)

    res_CE.append([param, 
                   accPathwaysCE[0], 
                   accPathwaysCE[1], 
                   accPhenoCE,
                   accBetaCE,
                   k,
                   repeat])
    
    res_additive.append([param,
                         accPhenoAdditive,
                         accBetaAdditive,
                         k,
                         repeat])
                         

    print("done with param %0.1f" %param)

with open(prefix + 'CE' + suffix + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res_CE: writer.writerow(row)
        
with open(prefix + 'additive' + suffix + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res_additive: writer.writerow(row)
        
