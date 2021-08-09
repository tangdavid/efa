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
prefix = './additive-var/'
suffix = '_k%d_repeat%d' % (k, repeat)

with open(prefix + 'training' + suffix + '.pkl', 'wb') as f:
    training = pkl.load(f)

with open(prefix + 'validation' + suffix + '.pkl', 'wb') as f:
    validation = pkl.load(f)

for param in training.key():
    data = training[param]
    oos = validation[param]
    min_loss = float('inf')
    pathways = None

    for restart in range(10):
        ceModel = CEModel()
        ceModel.gradDescent(data)
        loss = ceModel.loss[-1]
        if loss < min_loss:
            min_loss = loss
            pathways = ceModel.pathways           
            accPathwaysCE = ceModel.evalPathwayAcc(data)
            accPhenoCE = ceModel.evalPhenoAcc(data)
            accBetaCE = ceModel.evalPhenoAcc(data)
            
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
                         accPhenoCE,
                         accBetaCE,
                         k,
                         repeat])
                         

    print("done with h2 %0.1f" %h2)

with open(prefix + 'CE' + suffix + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res_CE: writer.writerow(row)
        
with open(prefix + 'additive' + suffix + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res_additive: writer.writerow(row)
        
