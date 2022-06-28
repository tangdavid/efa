import sys
sys.path.insert(0,'../model')

from models import UncoordinatedModel, CoordinatedModel, AdditiveModel
from tools import tools
from datasets import SimDataset
import pickle as pkl
import argparse
import csv


def benchmark(model_name, prefix, suffix, model_k, true_k):
    res = list()
    with open(prefix + 'training' + suffix + '.pkl', 'rb') as f:
        training = pkl.load(f)

    with open(prefix + 'validation' + suffix + '.pkl', 'rb') as f:
        validation = pkl.load(f)

    if model_k != 0:
        suffix = '_k%d%s' % (model_k, suffix)

    for param in training.keys():
        data = training[param]
        oos = validation[param]

        if model_name == 'coordinated':
            model = CoordinatedModel(k = model_k)
            model.fitModel(data, selfInteractions = False, anchors = False)
        elif model_name == 'coordinated_self':
            model = CoordinatedModel(k = model_k)
            model.fitModel(data, selfInteractions = True, anchors = True)
        elif model_name == 'uncoordinated':
            model = UncoordinatedModel()
            model.fitModel(data, random_effects=False)
        elif model_name == 'uncoordinated_random':
            model = UncoordinatedModel()
            model.fitModel(data, random_effects=True)
        elif model_name == 'additive':
            model = AdditiveModel()
            model.fitModel(data)
        else:
            print('poorly specified model')
            exit()

        accPheno = model.evalPhenoAcc(oos)
        accBeta = model.evalBetaAcc(data)
        accOmega = model.evalOmegaAcc(data)
        if 'coordinated' not in model_name or model_k != true_k:
            model.omegaPCA(k=true_k)
        accPathways = model.evalPathwayAcc(data)
        res.append([param, accPheno, accBeta, accOmega, accPathways[0], accPathways[1],  repeat])
                
        print("done with param %0.1f" %param)
        sys.stdout.flush()
            
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
    prefix = './%s_k%d_self/' % (args.param, true_k)
    suffix = '_repeat%d' % (repeat)
    model_name = args.model
    benchmark(model_name, prefix, suffix, model_k, true_k) 


