from additiveSim import dataset, decomp
import numpy as np
import time
import csv

data = list()

def timeMethods(ind, snps, method, k, h2, sbeta, somega):
    benchmark = decomp()
    ret = list()
    for i in range(15):
        simStart = time.time()
        benchmark.simData(ind, snps, k = k, h2 = h2, sbeta = sbeta, somega = somega)
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

        fit = benchmark.evalAcc()
        loss = benchmark.getLoss(benchmark.pathways, benchmark.weights)

        simTime = simEnd - simStart
        fitTime = end - start
        decompAcc = fit

        ret.append((ind, snps, k, h2, sbeta, somega,
                    simTime, fitTime, decompAcc, loss, method))
        
    return(ret)
    

for k in range(2, 6):
    for h2 in np.linspace(0.1, 0.9, 3):
        for sbeta  in np.linspace(0, 1, 3):
            for somega in np.linspace(0, 1, 3):
                for m in np.linspace(1, 2, 4):
                    n = 5
                    h2 = 0.9
                    ind = round(10 ** n)
                    snps = round(10 ** m)

                    print(k, h2, sbeta, somega, ind, snps)
                    data += (timeMethods(ind, snps, "direct", sbeta = sbeta, somega = somega, k = k, h2 = h2))
                break
            break
        break

with open('./benchmark.csv','w') as out:
    csvOut=csv.writer(out)
    csvOut.writerow(['n','m', 'k', 'h2', 'sbeta', 'somega', 
                     'simTime', 'fitTime', 'acc', 'loss', 'method'])
    for row in data:
        csvOut.writerow(row)
