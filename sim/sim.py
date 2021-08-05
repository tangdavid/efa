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
    
    def generateOOS(data, n):
        m, k, h2 = data.m, data.k, data.h2
        res = Dataset(n, m, k = k, h2 = h2)
        res.withEffectSizes(data)
        return res    
    
class Dataset:
    def __init__(self, n, m, k = 2,
                 h2 = 1, noise_beta = 0, noise_omega = 0, additive_model_var = 0.5,
                 sparse = 0, self_interactions = True, anchor_strength = 0.5):
        self.n = n
        self.m = m
        self.k = k
        self.h2 = h2
        self.sparse = sparse
        self.noise_beta = noise_beta
        self.noise_omega = noise_omega
        self.additive_model_var = additive_model_var
        self.self_interactions = self_interactions
        self.anchor_strength = anchor_strength

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

    def simEffects(self):
        m = self.m
        k = self.k

        # simulated latent pathways
        pathways = np.random.normal(0, 1, (m, k))
        sparse = np.random.uniform(0, 1, (m, k)) > self.sparse
        pathways = pathways * sparse

        # add in anchor snps
        pathways[-k:, :] = np.eye(k)
        
        norm_anchors = linalg.norm(pathways[-k:, :])
        norm_others = linalg.norm(pathways[:-k, :])

        anchor_strength = self.anchor_strength
        scale_num = anchor_strength * norm_others
        scale_den = (norm_anchors * (1 - anchor_strength) + anchor_strength * norm_others) 
        scale = scale_num / scale_den
        pathways[-k:, :] *= scale
        pathways[:-k, :] *= (1 - scale)
        

        # main effects as sum of pathways
        beta = np.sum(pathways, axis=1, keepdims=True)

        # generate weights from normal (0, 1)
        weights = np.random.normal(0, 1, size = (k, k))
        offset = 0 if self.self_interactions else -1 
        weights = np.tril(weights, offset) + np.tril(weights, -1).T

        # simulate interaction matrix by summing over weighted outerproducts
        omega = tools.outer(pathways, weights)

        # adding gaussian noise to additive and epistatic effects
        var_eomega = np.var(omega) * (self.noise_omega/ (1 - self.noise_omega)) 
        eomega = np.random.normal(0, np.sqrt(var_eomega), m * m).reshape(m, -1)
        eomega = np.tril(eomega) + np.tril(eomega, -1).T
        
        var_ebeta = np.var(beta) * (self.noise_beta/ (1 - self.noise_beta)) 
        ebeta = np.random.normal(0, np.sqrt(var_ebeta), m).reshape(-1, 1)
        
        # scale additive and epistatic effect sizes
        e_var = np.var(self.inter @ (omega + eomega).reshape(-1, 1))
        a_var = np.var(self.geno @ (beta + ebeta))
        
        scale = np.sqrt((a_var/self.additive_model_var - a_var)/e_var)
        weights *= scale
        eomega *= scale
        omega = tools.outer(pathways, weights)
        
        # instance variables
        self.pathways = pathways
        self.beta = beta + ebeta
        self.omegaMat = omega + eomega
        self.omega = self.omegaMat.reshape(-1, 1)
        self.weights = weights
        self.normalizePheno()
        
    
    def normalizePheno(self):
        # normalize phenotypes to variance 1
        std_mean = np.std(self.inter @ self.omega + self.geno @ self.beta)
        scale = std_mean / np.sqrt(self.h2)
        self.pathways /= scale
        self.weights *= scale
        self.beta /= scale
        self.omegaMat /= scale
        
    def withEffectSizes(self, data):
        self.pathways = data.pathways
        self.weights = data.weights
        self.beta = data.beta
        self.omega = data.omega
        self.omegaMat = data.omegaMat
        self.simPheno()

    def simPheno(self):
        # model with main effects and interactions
        mean = self.inter @ self.omega + self.geno @ self.beta

        # add noise to simulate heritability
        eps_std = np.sqrt(1 - self.h2)
        noise = np.random.normal(0, eps_std, self.n).reshape(-1, 1)
        self.pheno = mean + noise
                
class Decomp:
    def __init__(self):
        self.loss = None
        self.conv = True
        self.pathways = None
        self.weights = None
    
    def coordDescent(self, data, progress = False):
        m = data.m
        k = data.k

        G = data.geno
        inter = data.inter
        Y = data.pheno#.reshape(-1, 1)

        tol = 1e-5

        # initialize pathways and weights
        weights = np.random.normal(0, 0.1, size = (k, k))
        weights = np.tril(weights, -1) + np.tril(weights, -1).T
        pathways = np.random.normal(0, 0.1, size = (m, k))
        iterations = 0

        prevLoss = currentLoss = self.getLoss(data, pathways, weights, tensor = False)
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
                u[m-k+i] = 0
                pathways[:, i] = u.reshape(-1,)

            weights = self.getWeights(data, pathways, diag = False)

            # monitor convergence
            iterations += 1

            prevLoss = currentLoss
            currentLoss = self.getLoss(data, pathways, weights, tensor = False)
            lossList.append(currentLoss)

            if np.abs(currentLoss - prevLoss) < tol: break
            if iterations > 10000: break
            if progress and iterations % 100 == 0:
                print("(iterations,loss):", iterations, round(currentLoss, 3))

        self.loss = lossList
        self.pathways = pathways
        self.weights = weights

    def gradDescent(self, data, reg = 0, progress = False, analyticalWeights = False):
        m = data.m
        k = data.k

        G = torch.tensor(data.geno, requires_grad = False).double()
        inter = torch.tensor(data.inter, requires_grad = False).double()
        Y = torch.tensor(data.pheno, requires_grad = False).double()

        tol = 1e-4
        
        # for anchor snps
        mask = torch.ones(size = (m, k))
        mask[-k:, :] = torch.eye(k)
        
        # initialization
        weights = np.random.normal(0, 0.1, size = (k, k))
        weights = np.tril(weights, -1) + np.tril(weights, -1).T
        weights = torch.tensor(weights, requires_grad = True)

        pathways = np.random.normal(0, 0.1, size = (m, k))
        pathways *= mask.detach().numpy()
        pathways = torch.tensor(pathways, requires_grad = True)
        
        # use Adam to optimize 
        params = [pathways] if analyticalWeights else [weights, pathways]
        optimizer = optim.Adam(params)
        prevLoss = currentLoss = self.getLoss(data, pathways, weights).item()
        
        lossList = [currentLoss]
        iterations = 0

        conv = True
        
        while True:            
            iterations += 1
            
            if analyticalWeights: 
                weights = torch.tensor(self.getWeights(data, pathways.detach().numpy()))
            loss = self.getLoss(data, pathways * mask, weights, reg = reg)
            
            # optimize the loss function with gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # monitor convergence
            prevLoss = currentLoss
            currentLoss = loss.item()
            lossList.append(currentLoss)
            
            if np.abs(currentLoss - prevLoss)/prevLoss < tol and iterations > 1: break
            if iterations % 1000 == 0 and progress: 
                print("(iterations,loss):", iterations, round(loss.item(), 3))
                
            if iterations > 15000: break

        pathways = pathways.detach().numpy()
        weights = weights.detach().numpy()
        
        self.loss = lossList
        self.pathways = pathways
        self.weights = weights
        self.conv = conv
        
    def getWeights(self, data, pathways, diag = True):
        G = data.geno
        Y = data.pheno
        k = data.k
        
        beta = pathways.sum(axis = 1, keepdims = True)
        
        Z = Y - G @ beta
        X = linalg.khatri_rao((G @ pathways).T, (G @ pathways).T).T
        
        offset = 0 if diag else -1
        
        indices = np.nonzero(np.tril(np.ones(k), offset).reshape(-1,))[0]
        X = X[:, indices]

        lr = LinearRegression(fit_intercept=True)
        lr.fit(X, Z)
        
        weightsTril = lr.coef_.reshape(-1,)
        weights = np.zeros((k, k))
        weights[np.tril_indices(k, offset)] = weightsTril
        weights[np.tril_indices(k, -1)] *= 1/2
        weights = weights + np.tril(weights, -1).T
        return weights

        
        
    def getLoss(self, data, pathways, weights, reg = 0, tensor = True):
        G = torch.tensor(data.geno) if tensor else data.geno
        Y = torch.tensor(data.pheno) if tensor else data.pheno
        inter = torch.tensor(data.inter) if tensor else data.inter
        n = data.n

        if tensor:
            interEffect = tools.outer(pathways, weights, tensor=tensor).view(-1, 1).double()
            interEffect = inter @ interEffect
            mainEffect = torch.sum(G @ pathways, dim=1, keepdims=True)
            penalty = reg * torch.sum(torch.abs(pathways))
            loss = torch.norm(Y - mainEffect - interEffect)**2 + reg
        else:
            interEffect = tools.outer(pathways, weights, tensor=tensor).reshape(-1, 1)
            interEffect = inter @ interEffect
            mainEffect = np.sum(G @ pathways, axis=1, keepdims=True)
            penalty = reg * np.sum(np.abs(pathways))
            loss = linalg.norm(Y - mainEffect - interEffect)**2 + reg
        
        return loss
    
    def evalPathwayAcc(self, data):
        k = data.k
        withAnchors = tools.evalAcc(self.pathways, data.pathways)
        withoutAnchors = tools.evalAcc(self.pathways[:-k], data.pathways[:-k])
        return (withAnchors, withoutAnchors)
    
    def evalPhenoAcc(self, data):
        prediction = self.predictPheno(data)
        return stats.pearsonr(prediction.reshape(-1,), data.pheno.reshape(-1,))[0] ** 2
    
    def predictPheno(self, data):
        beta = np.sum(self.pathways, axis = 1, keepdims=True)
        omega = tools.outer(self.pathways, self.weights)
        return data.geno @ beta + data.inter @ omega.reshape(-1, 1)
        
    
    def plotLoss(self):
        plt.plot(self.loss)
        plt.show()