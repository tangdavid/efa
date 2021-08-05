from sim import *
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument("k", help="number of latent pathways", type=int)
parser.add_argument("repeat", help="index of repeat", type=int)
args = parser.parse_args()

res = list()
k = args.k
repeat = args.repeat
name = 'anchor_strength_k%d_repeat%d' % (k, repeat)
print(name)

for anchor_strength in np.linspace(0, 0.9, 10):
    data = Dataset(10000, 20, k = k, h2=0.7, additive_model_var=0.5, anchor_strength=anchor_strength)
    oos = tools.generateOOS(data, 10000)
    min_loss = float('inf')
    pathways = None

    for restart in range(10):
        decomp = Decomp()
        decomp.gradDescent(data)
        loss = decomp.loss[-1]
        if loss < min_loss:
            min_loss = loss
            pathways = decomp.pathways           
            accPathways = decomp.evalPathwayAcc(data)
            accPheno = decomp.evalPhenoAcc(data)
            

    res.append([np.round(anchor_strength, 1), 
                accPathways[0], 
                accPathways[1], 
                accPheno,
                k,
                repeat])

    print("done with anchor_strength %0.1f" %anchor_strength)

with open('./anchor-strength/' + name + '.csv', 'w') as f:
    writer = csv.writer(csvfile)
    for row in res:
        writer.writerow(row)
        
