from additiveSim import dataset, decomp
import numpy as np
import time
import csv

results = []

for k in range(3, 5):
    for m in np.linspace(1, 1, 1):
        for i in range(1):
            snps = round(10 ** m)

            test = decomp()
            test.simData(10000, snps, k=k, h2=0.999)

            for j in range(30):
                print(k, snps, j)
                start = time.time()
                (loss, iterations, pathways, weights) = test.coordDescent()
                end = time.time()

                coordTime = end - start
                test.pathways = pathways
               
                ret = ['coord'] + [k] + [snps] + [j] + [test.evalAcc()] + [coordTime] + loss
                results.append(ret)

                #start = time.time()
                #(loss, iterations, pathways, weights) = test.gradDescent()
                #end = time.time()

                #gradTime = end - start
                #test.pathways = pathways
               
                #ret = ['grad'] + [k] + [snps] + [j] + [test.evalAcc()] + [gradTime] + loss
                #results.append(ret)
            break
        break
    break

with open('./convergence.csv','w') as out:
    csvOut=csv.writer(out)
    #csvOut.writerow(['k', 'm', 'rep', 'init', 'initLoss', 'finalLoss', 'iter', 'acc', 'time', 'method'])
    for row in results:
        csvOut.writerow(row)
