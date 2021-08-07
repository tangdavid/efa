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
name = 'h2_k%d_repeat%d' % (k, repeat)
print(name)

for h2 in np.linspace(0.1, 1, 10):
    data = Dataset(10000, 20, k = k, h2=h2)
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
            

    res.append([np.round(h2, 1), 
                accPathways[0], 
                accPathways[1], 
                accPheno,
                k,
                repeat])

    print("done with h2 %0.1f" %h2)

with open('./h2/' + name + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res:
        writer.writerow(row)
        
