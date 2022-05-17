import numpy as np
from scipy import linalg, stats
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import decomposition
from tqdm.notebook import tqdm, trange
import time
import itertools
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle as pkl


class tools:
    def outer(mat, weights, tensor = False, skip = None):
        (m, k) = mat.shape
        matType = torch if tensor else np
        ret = torch.zeros(m, m) if tensor else np.zeros(shape = (m, m))
        
        for i in range(k):
            for j in range(k):
                if i == skip or j == skip:
                    continue
                    
                u = mat[:, i]
                v = mat[:, j]
                ret += weights[i, j] * matType.outer(u, v)
                
        return ret
    
    def face_splitting(t1, t2):
        dim = t1.shape[1] * t2.shape[1]
        return torch.bmm(t1.unsqueeze(2), t2.unsqueeze(1)).view(-1, dim)
    
    def evalAcc(p1, p2):
        maxCorr = 0
        if p1.shape[1] != p2.shape[1]:
            print('error: wrong dimensions')
            return
        
        k = p1.shape[1]

        for perm in itertools.permutations(range(k)):
            p1_perm = p1[:,perm].reshape(-1,)
            r = stats.pearsonr(p1_perm, p2.reshape(-1,))[0]
            rsquared = r ** 2
            if rsquared > maxCorr: 
                maxCorr = rsquared
                best_perm = perm

        return maxCorr
    
    
    def likelihoodRatioTest(data):
        k = data.k
        coordinated = CoordinatedModel(k)
        coordinated.fitModel(data, additiveInit = True)

        null = AdditiveModel()
        null.fitModel(data)

        coordinated.predictPheno
        coordinated_loss = coordinated.getLoss(data, coordinated.pathways, coordinated.weights, tensor = False)
        null_loss = linalg.norm(data.pheno - data.geno @ null.beta) ** 2

        sigma2 = np.var(data.pheno - coordinated.predictPheno(data))
        stat = max(1/sigma2 * (null_loss - coordinated_loss), 0)
        df = data.beta.shape[0] * (k - 1) + k * (k-1)/2 + k
        
        pval = 1 - stats.chi2.cdf(stat, df = df)
        return pval
    
    def computeLoss(geno, pheno, pathways, weights):
        G = geno
        Y = pheno
        n = data.n

        GU = G @ pathways
        interEffect = tools.face_splitting(GU, GU) @ weights.view(-1, 1)
        mainEffect = torch.sum(G @ pathways, dim=1, keepdims=True)
        penalty = reg * torch.sum(torch.abs(pathways))
        loss = torch.norm(Y - mainEffect - interEffect)**2
