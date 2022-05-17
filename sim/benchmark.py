import sys
sys.path.insert(0,'../model')

from models import *
from tools import *
from datasets import *
import argparse
import csv


def benchmark(model_name, prefix, suffix, model_k, true_k):
    res = list()
    with open(prefix + 'training' + suffix + '.pkl', 'rb') as f:
        training = pkl.load(f)

    with open(prefix + 'validation' + suffix + '.pkl', 'rb') as f:
        validation = pkl.load(f)

    suffix = '_k%d%s' % (model_k, suffix)

    for param in training.keys():
        data = training[param]
        oos = validation[param]

        if model_name == 'coordinated':
            model = CoordinatedModel(k = model_k)
            model.fitModel(data)
            accPheno = model.evalPhenoAcc(oos)
            accBeta = model.evalBetaAcc(data)
            accOmega = model.evalOmegaAcc(data)
            if model_k == true_k: 
                accPathways = model.evalPathwayAcc(data)
                res.append([param, accPheno, accBeta, accOmega, accPathways[0], accPathways[1], model_k, repeat])
            else:
                res.append([param, accPheno, accBeta, accOmega, model_k, repeat])
        elif model_name == 'coordinated_self':
            model = CoordinatedModel(k = model_k)
            model.fitModel(data, selfInteractions = True)
            accPheno = model.evalPhenoAcc(oos)
            accBeta = model.evalBetaAcc(data)
            accOmega = model.evalOmegaAcc(data)
            if model_k == true_k: 
                accPathways = model.evalPathwayAcc(data)
                res.append([param, accPheno, accBeta, accOmega, accPathways[0], accPathways[1], model_k, repeat])
            else:
                res.append([param, accPheno, accBeta, accOmega, model_k, repeat])
        elif model_name == 'uncoordinated':
            model = UncoordinatedModel()
            model.fitModel(data)
            accPheno = model.evalPhenoAcc(oos)
            accBeta = model.evalBetaAcc(data)
            accOmega = model.evalOmegaAcc(data)
            res.append([param, accPheno, accBeta, accOmega, model_k, repeat])
        elif model_name == 'additive':
            model = AdditiveModel()
            model.fitModel(data)
            accPheno = model.evalPhenoAcc(oos)
            accBeta = model.evalBetaAcc(data)
            res.append([param, accPheno, accBeta, model_k, repeat])
        else:
            print('poorly specified model')
            exit()
                
        print("done with param %0.1f" %param)
            
    with open(prefix + model_name + suffix + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in res: writer.writerow(row)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kmodel", help="number of latent pathways", type=int)
    parser.add_argument("ktrue", help="number of latent pathways", type=int)
    parser.add_argument("repeat", help="index of repeat", type=int)
    parser.add_argument("param", help="which param to benchmark", type=str)
    parser.add_argument("model", help="which model to fit", type=str)
    args = parser.parse_args()

    model_k = args.kmodel
    true_k = args.ktrue
    repeat = args.repeat
    prefix = './%s_k%d/' % (args.param, true_k)
    suffix = '_repeat%d' % (repeat)
    model_name = args.model
    benchmark(model_name, prefix, suffix, model_k, true_k) 


