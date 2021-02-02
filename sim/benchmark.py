from sim import dataset, decomp
import numpy as np
import time
import csv

data = list()

def timeMethods(ind, snps, method, k = 1, h2 = 0.9, noise = 0):
    benchmark = decomp()
    ret = list()
    for i in range(10):
        simStart = time.time()
        benchmark.simData(ind, snps, k = k, h2 = h2, noise = noise)
        simEnd = time.time()

        if method == "direct":
            start = time.time()
            benchmark.directDecomp()
            end = time.time()
        elif method == "marginal":
            start = time.time()
            benchmark.fitMarginal()
            benchmark.symmetricDecomp()
            end = time.time()
        else:
            start = time.time()
            benchmark.fitRidge()
            benchmark.symmetricDecomp()
            end = time.time()

        _, ufit, vfit = benchmark.evalNorm()

        simTime = simEnd - simStart
        fitTime = end - start
        decompAcc = ufit

        ret.append((ind, snps, simTime, fitTime, decompAcc, method))
        
    return(ret)
    

for noise  in np.linspace(0, 1, 5):
    for m in np.linspace(1, 2, 5):

        n = 5

        ind = round(10 ** n)
        snps = round(10 ** m)

        print(noise, ind, snps)

        data += (timeMethods(ind, snps, "ridge", noise = noise, k = 1, h2 = 0.9))
        data += (timeMethods(ind, snps, "marginal", noise = noise, k = 1, h2 = 0.9))
        data += (timeMethods(ind, snps, "direct", noise = noise, k = 1, h2 = 0.9))

with open('./benchmark.csv','w') as out:
    csvOut=csv.writer(out)
    csvOut.writerow(['n','m', 'simTime', 'fitTime', 'acc', 'method'])
    for row in data:
        csvOut.writerow(row)
