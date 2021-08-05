from sim import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("k", help="number of latent pathways", type=int)
parser.add_argument("repeat", help="index of repeat", type=int)
args = parser.parse_args()

res = dict()
k = args.k
repeat = args.repeat
name = 'anchor_strength_k%d_repeat%d' % (k, repeat)
print(name)

for anchor_strength in np.linspace(0, 0.9, 10):
    data = Dataset(10000, 20, k = k, h2=0.7, additive_model_var=0.5, anchor_strength=anchor_strength)
    oos = tools.generateOOS(data, 10000)
    min_loss = np.float('inf')
    pathways = None

    for restart in range(1):
        decomp = Decomp()
        decomp.gradDescent(data)
        loss = decomp.loss[-1]
        if loss < min_loss:
            min_loss = loss
            pathways = decomp.pathways           
            accPathways = decomp.evalPathwayAcc(data)
            accPheno = decomp.evalPhenoAcc(data)
            
    res[np.round(anchor_strength, 1)] = (accPathways[0], accPathways[1], accPheno)

with open('./anchor/' + name + 'pkl', 'wb') as f:
    pkl.dump(res, f)
        