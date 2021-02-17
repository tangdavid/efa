import numpy as np
from scipy import linalg, stats
from sklearn import linear_model
from sklearn import decomposition
import time
import itertools
import torch
import torch.optim as optim

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


class dataset:
    def __init__(self, n, m, k, h2, sbeta, somega):
        self.n = n
        self.m = m
        self.k = k
        self.h2 = h2
        self.sbeta = sbeta
        self.somega = somega

        self.simGeno()
        self.simEffects()
        self.simPheno()

    def simGeno(self):
        # genotypes are iid binom(2, p) where p normal
        # genotypes are scaled and centered

        geno = np.zeros([self.n, self.m])
        for i in range(self.m):
            p = np.random.beta(2, 2)
            snps = np.random.binomial(2, p, self.n)
            geno.T[i] = (snps - (2*p))/np.sqrt(2*p*(1-p))

        # interaction effects as khatri rao
        inter = linalg.khatri_rao(geno.T, geno.T).T
        self.geno, self.inter = geno, inter

    def simEffects(self, selfInteraction = False):
        m = self.m
        k = self.k

        # simulated latent pathways
        pathways = np.random.normal(0, 1, m * k).reshape(m, -1)

        # main effects as sum of pathways
        beta = np.sum(pathways, axis=1, keepdims=True)

        # currently generate weights from normal (0, 1)
        # weights = np.ones(k*k).reshape(k, -1)
        weights = np.random.normal(0, 1, k * k).reshape(k, -1)
        if selfInteraction:
            weights = np.tril(weights, 1) + np.tril(weights, -1).T
        else:
            weights = np.tril(weights, -1) + np.tril(weights, -1).T

        # simulate interaction matrix by summing over weighted outerproducts
        omega = tools.outer(pathways, weights)

        # adding gaussian noise to interaction effects
        eomega = np.random.normal(0, self.somega, m * m).reshape(m, -1)
        eomega = np.tril(eomega) + np.tril(eomega, -1).T

        # adding gaussian noise to main effects
        ebeta = np.random.normal(0, self.sbeta, m).reshape(-1, 1)

        # instance variables
        self.pathways = pathways
        self.beta = beta + ebeta
        self.omegaMat = omega + eomega
        self.omega = self.omegaMat.reshape(-1, 1)
        self.omegaWeight = weights

    def simPheno(self):
        # model with main effects and interactions
        mean = self.inter @ self.omega + self.geno @ self.beta

        # add noise to simulate heritability
        var = np.var(mean) * (1 - self.h2)/self.h2
        sd = np.sqrt(var)
        noise = np.random.normal(0, sd, self.n).reshape(-1, 1)
        self.pheno = mean + noise
 

class decomp:
    def __init__(self):
        self.data = None
    def simData(self, n, m, k = 2, h2 = 0.9, sbeta = 0, somega = 0):
        self.data = dataset(n, m, k, h2, sbeta, somega)
        self.k = k
    
    def coordDescent(self):
        m = self.data.m
        k = self.k
        
        G = self.data.geno
        inter = self.data.inter
        Y = self.data.pheno#.reshape(-1, 1)
        
        thresh = 0.005
                
        # initialize pathways and weights
        weights = np.random.normal(0, 1, k * k).reshape(k, -1) 
        #weights = np.ones(k * k).reshape(k, -1) 
        weights = np.tril(weights, -1) + np.tril(weights, -1).T
        pathways = np.random.normal(0, 1/10, m * k).reshape(m, k)
        iterations = 0
        
        prevLoss = currentLoss = self.getLoss(pathways, weights)
        lossList = [currentLoss]
        
        while(True):
            # iteratively update each pathway
            for i in range(k):
                # compute constants
                mask = [True] * k
                mask[i] = False
                C1 = tools.outer(pathways, weights, skip=i).reshape(-1, 1)
                C1 = inter @ C1
                C2 = G @ np.sum(pathways[:,mask], axis=1, keepdims=True) 
                C = C1 + C2
                A = np.sum(G @ pathways @ np.diagflat(weights[i]), axis=1, keepdims=True)
                
                # equations from the first order conditions
                u1 = (4 * G.T @ (A*A*G)) + (4 * G.T @ (A*G)) + (G.T @ G)
                u2 = (2 * (A*G).T @ Y) + (G.T @ Y) - (2 * (A*G).T @ C) - (G.T @ C)
                u = linalg.inv(u1) @ u2
                pathways[:, i] = u.reshape(-1,)
                
                
            # iteratively fit weights
            for i in range(k):
                for j in range(i):
                    ui = pathways[:,i]
                    uj = pathways[:,j]
                    
                    # compute constants
                    interacting = ((G @ ui) * (G @ uj)).reshape(-1, 1)
                    C1 = tools.outer(pathways, weights) 
                    C1 = C1 - weights[i, j] * (np.outer(ui, uj) + np.outer(uj, ui))
                    C1 = inter @ C1.reshape(-1, 1)
                    C2 = G @ np.sum(pathways, axis=1, keepdims=True) 
                    C = C1 + C2
                    
                    # equation from the first order conditions
                    weight = 1/2 * (interacting.T @ (Y - C))/(interacting.T @ interacting)
                    weights[i, j] = weight
                    weights[j, i] = weight    
            
            # monitor convergence
            iterations += 1
            
            prevLoss = currentLoss
            currentLoss = self.getLoss(pathways, weights)
            lossList.append(currentLoss)
            
            if np.abs(currentLoss - prevLoss) < thresh: break
            #if iterations % 10 == 0:
            #    print("(iterations,loss):", iterations, round(currentLoss, 3))
            
        return (lossList, iterations, pathways, weights)
        
    def gradDescent(self):
        m = self.data.m
        k = self.k

        G = torch.tensor(self.data.geno, requires_grad = False).double()
        inter = torch.tensor(self.data.inter, requires_grad = False).double()
        Y = torch.tensor(self.data.pheno, requires_grad = False).double()

        thresh = 0.001
        
        # initialization
        #weights = np.ones(k * k).reshape(k, -1) 
        
        weights = np.random.normal(0, 1, k * k).reshape(k, -1) 
        weights = np.tril(weights, 0) + np.tril(weights, -1).T
        weights = torch.tensor(weights, requires_grad = True)

        pathways = torch.randn(m, k, requires_grad = True, dtype=torch.float64)
        
        # use Adam to optimize 
        optimizer = optim.Adam([weights, pathways])
        prevLoss = currentLoss = self.getLoss(pathways.detach().numpy(),
                                              weights.detach().numpy())
        
        lossList = [currentLoss]
        iterations = 0
        while True:
            iterations += 1
            # compute effects
            interEffect = tools.outer(pathways, weights, tensor = True).view(-1, 1).double()
            interEffect = inter @ interEffect
            mainEffect = torch.sum(G @ pathways, dim = 1, keepdim = True)
            
            # compute loss using effects
            optimizer.zero_grad()
            loss = torch.norm(Y - mainEffect - interEffect)
            
            # optimize the loss function with gradient descent
            loss.backward()
            optimizer.step()
            
            # monitor convergence
            prevLoss = currentLoss
            currentLoss = loss.item()
            lossList.append(currentLoss)

            if np.abs(currentLoss - prevLoss) < thresh: break
            #if iterations % 1000 == 0: 
            #    print("(iterations,loss):", iterations, round(currentLoss, 3))
            
        pathways = pathways.detach().numpy()
        weights = weights.detach().numpy()
        return (lossList, iterations, pathways, weights)
        
    def getLoss(self, pathways, weights):
        G = self.data.geno
        Y = self.data.pheno
        inter = self.data.inter
        
        interEffect = tools.outer(pathways, weights).reshape(-1, 1)
        interEffect = inter @ interEffect
        mainEffect = np.sum(G @ pathways, axis=1, keepdims=True)
        loss = linalg.norm(Y - mainEffect - interEffect)
        return(loss)
    
    def evalAcc(self):
        maxCorr = 0
        
        groundTruth = self.data.pathways.reshape(-1,)
        for perm in itertools.permutations(range(self.k)):
            prediction = self.pathways[:,perm].reshape(-1,)
            r = stats.pearsonr(prediction, groundTruth)[0]
            rsquared = r ** 2
            if rsquared > maxCorr: 
                maxCorr = rsquared
        
        return(maxCorr)
            
                
