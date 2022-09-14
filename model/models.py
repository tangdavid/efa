import numpy as np
from tools import tools
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


from scipy.stats import pearsonr
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.linalg import khatri_rao, norm

import torch
from torch import optim

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

            

class Model(ABC):
    def omegaPCA(self, k=2):
        omega_mat = self.omega.reshape(self.m, self.m)
        pca = PCA(n_components=k)
        pca.fit(omega_mat)
        self.pathways = pca.components_.T

    def evalPathwayAcc(self, data):
        k = data.k
        withAnchors = tools.evalAcc(self.pathways, data.pathways)
        withoutAnchors = tools.evalAcc(self.pathways[:-k], data.pathways[:-k])
        return (withAnchors, withoutAnchors)
    
    def evalBetaAcc(self, data):
        return pearsonr(self.beta.reshape(-1,), data.beta.reshape(-1,))[0] ** 2
    
    def evalOmegaAcc(self, data):
        return pearsonr(self.omega.reshape(-1,), data.omega.reshape(-1,))[0] ** 2
    
    def evalPhenoAcc(self, data):
        prediction = self.predictPheno(data)
        return pearsonr(prediction.reshape(-1,), data.pheno.reshape(-1,))[0] ** 2

    @abstractmethod
    def fitModel(self, data, *argv, **kwargs):
        pass

    @abstractmethod
    def predictPheno(self, data):
        pass

class CoordinatedModel(Model):
    def __init__(self, k):
        self.k = k
        self.loss = None
        self.conv = True
        self.pathways = None
        self.weights = None

    def gradDescent(self, data, reg, progress,
                    nullModel, additiveInit, selfInteractions, anchors):
        m = data.m
        k = self.k

        tol = 1e-7
        res = dict()
        
        weights, pathways, pathways_mask = self.initPathways(data, nullModel, additiveInit)
        
        # use Adam to optimize 
        params = [pathways] if nullModel else [weights, pathways]
        optimizer = optim.Adam(params)
        prevLoss = currentLoss = self.getLoss(data, pathways, weights).item()
        
        lossList = [currentLoss]
        iterations = 0

        conv = True
        
        weights_mask = torch.ones(size = (k, k))
        if not selfInteractions: weights_mask = weights_mask - torch.eye(k)
            
        if not anchors: pathways_mask = torch.ones(size = (m, k))
        
        while True:            
            iterations += 1
            
            loss = self.getLoss(data, pathways * pathways_mask, weights * weights_mask, reg = reg)
            
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
        
        
        res['loss'] = lossList
        res['pathways'] = pathways
        res['beta'] = np.sum(pathways, axis = 1, keepdims=True)
        res['weights'] = weights
        res['omega'] = tools.outer(pathways, weights).reshape(-1, 1)
        res['conv'] = conv
        return res
        
    def initPathways(self, data, nullModel, additiveInit):
        m = data.m
        k = self.k
        
        # for anchor snps
        mask = torch.ones(size = (m, k))
        mask[-k:, :] = torch.eye(k)
        
        # initialization
        weights = np.random.normal(0, 0.1, size = (k, k))
        weights = np.tril(weights, -1) + np.tril(weights, -1).T
            
        pathways = np.random.normal(0, 0.1, size = (m, k))
        pathways *= mask.detach().numpy()
        weights = torch.tensor(weights, requires_grad = True)            
        
        if nullModel: weights = torch.zeros(k, k)
        
        if additiveInit:
            lr = LinearRegression()
            lr.fit(data.geno, data.pheno)
            additive = lr.coef_.reshape(-1, 1)
            pathways[:, -1:] = additive - pathways[:, :-1].sum(axis = 1, keepdims = True)
            pathways[-k:, :] = np.diagflat(additive[-k:])
            

        pathways = torch.tensor(pathways, requires_grad = True)
        
        return weights, pathways, mask
        
    def getWeights(self, data, pathways, diag = True):
        G = data.geno
        Y = data.pheno
        k = self.k
        
        beta = pathways.sum(axis = 1, keepdims = True)
        
        Z = Y - G @ beta
        X = khatri_rao((G @ pathways).T, (G @ pathways).T).T
        
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
        G = torch.tensor(data.geno).double() if tensor else data.geno
        Y = torch.tensor(data.pheno).double() if tensor else data.pheno
        n = data.n

        if tensor:
            GU = G @ pathways
            interEffect = tools.face_splitting(GU, GU) @ weights.view(-1, 1)
            mainEffect = torch.sum(G @ pathways, dim=1, keepdims=True)
            penalty = reg * torch.sum(torch.abs(pathways))
            loss = torch.norm(Y - mainEffect - interEffect)**2 + penalty
        else:
            GU = G @ pathways
            interEffect = khatri_rao(GU.T, GU.T).T @ weights.reshape(-1, 1)
            mainEffect = np.sum(G @ pathways, axis=1, keepdims=True)
            penalty = reg * np.sum(np.abs(pathways))
            loss = norm(Y - mainEffect - interEffect)**2 + penalty
        
        return loss

    def plotLoss(self):
        plt.plot(self.loss)
        plt.show()

    def fitModel(self, data, reg = 0, restarts = 10, progress = False, 
                 nullModel = False, additiveInit = False, selfInteractions = True, anchors = True):
        minLoss = float('inf')
        minLossRes = None
        for restart in range(restarts):
            res = self.gradDescent(data, reg, progress, 
                                   nullModel, additiveInit, selfInteractions, anchors)
            if res['loss'][-1] < minLoss:
                minLoss = res['loss'][-1]
                minLossRes = res
        
        self.m = data.m
        self.loss = minLossRes['loss']
        self.pathways = minLossRes['pathways']
        self.beta = minLossRes['beta']
        self.weights = minLossRes['weights']
        self.omega = minLossRes['omega']
        self.conv = minLossRes['conv']
    
    def predictPheno(self, data):
        G = data.geno
        U = self.pathways
        weights = self.weights
        
        GU = G @ U
        interEffect = khatri_rao(GU.T, GU.T).T @ weights.reshape(-1, 1)
        mainEffect = np.sum(GU, axis=1, keepdims=True)
        
        return mainEffect + interEffect
        
class AdditiveModel(Model):
    def evalOmegaAcc(self, data):
        return None
    
    def evalPathwayAcc(self, data):
        return (None, None)
    
    def omegaPCA(self, k):
        pass
    
    def NLL(self, var, data):
        G = data.geno
        Y = data.pheno.reshape(-1,)
        m = data.m
        n = data.n
        K = 1/m * G @ G.T
        V = var[0] * K + var[1] * np.eye(n)
        mu = np.zeros(n)
        negloglike = -multivariate_normal.logpdf(Y, mu, V, allow_singular=True)
        return negloglike

    def fitMLE(self, data):
        bounds = [(1e-6, None)] * 2
        optim = minimize(self.NLL, [1, 1], data, bounds=bounds)    

        G = data.geno
        Y = data.pheno

        self.var = optim['x']
        self.G = G
        self.Y = Y
        
        return optim['x']
    
    def fitBLUP(self, data):
        G = data.geno
        Y = data.pheno
        m_G = G.shape[1]
        K = 1/m_G * G @ G.T
        n = data.n

        var = self.fitMLE(data)
        V = var[0] * K + var[1] * np.eye(n)

        beta = (var[0]/m_G * G).T @ np.linalg.inv(V) @ Y

        return beta

    def fitModel(self, data, random_effects=False):
        if random_effects:
            beta = self.fitBLUP(data)
        else:
            lr = LinearRegression()
            lr.fit(data.geno, data.pheno)
            beta = lr.coef_.reshape(-1, 1)
        self.beta = beta

    def predictPheno(self, data):
        if hasattr(self, 'beta'):
            return data.geno @ self.beta
        if hasattr(self, 'var'):
            return self.imputePheno(data)
        else:
            print('error: need to fit either effect sizes or var components')
            return 
    
    def imputePheno(self, data):
        Yin = self.Y
        m = data.m
        Kin = 1/m * self.G @ self.G.T
        Kout = 1/m * data.geno @ self.G.T
        var = self.var
        Vin = var[0] * Kin + var[1] * np.eye(self.G.shape[0])
        Vout = var[0] * Kout
        return Vout @ np.linalg.inv(Vin) @ Yin

class UncoordinatedModel(Model):
    def NLL(self, var, data):
        G = data.geno
        Y = data.pheno.reshape(-1,)
        m = data.m
        n = data.n
        K = 1/m * G @ G.T
        V = var[0] * K + var[1] * K * K + var[2] * np.eye(n)
        mu = np.zeros(n)
        negloglike = -multivariate_normal.logpdf(Y, mu, V, allow_singular=True)
        return negloglike

    def fitMLE(self, data):
        bounds = [(1e-6, None)] * 3
        optim = minimize(self.NLL, [1, 1, 1], data, bounds=bounds)    

        G = data.geno
        Y = data.pheno

        self.var = optim['x']
        self.G = G
        self.Y = Y
        return optim['x']
    
    def fitBLUP(self, data):
        G = data.geno
        Y = data.pheno
        GG = khatri_rao(G.T, G.T).T
        m_G = G.shape[1]
        m_GG = GG.shape[1]
        K = 1/m_G * G @ G.T
        n = data.n

        var = self.fitMLE(data)
        V = var[0] * K + var[1] * K * K + var[2] * np.eye(n)

        beta = (var[0]/m_G * G).T @ np.linalg.inv(V) @ Y
        omega = (var[1]/m_GG * GG).T @ np.linalg.inv(V) @ Y

        return (beta, omega)

    def fitBeta(self, data):
        lr = LinearRegression()
        lr.fit(data.geno, data.pheno)
        self.beta = lr.coef_.reshape(-1, 1)
        
    def fitOmega(self, data):
        lr = LinearRegression(fit_intercept=True)
        
        Y = data.pheno - data.geno @ self.beta
        inter = khatri_rao(data.geno.T, data.geno.T).T
        
        omega = np.zeros(data.m ** 2)
        for i in range(data.m):
            for j in range(data.m):
                idx = i * data.m + j    
                interij = inter.T[idx]
                regressors = interij.reshape(-1, 1)
                lr.fit(regressors, Y)
                omega[idx] = lr.coef_.reshape(-1,)[0]
        self.omega = omega.reshape(-1, 1)
    
    def fitModel(self, data, random_effects = False):
        if random_effects:
            beta, omega = self.fitBLUP(data)
            self.beta = beta
            self.omega = omega
        else:
            self.fitBeta(data)
            self.fitOmega(data)
        self.m = data.m
        self.n = data.n
        
    def predictPheno(self, data):
        if hasattr(self, 'beta') and hasattr(self, 'omega'):
            inter = khatri_rao(data.geno.T, data.geno.T).T
            return data.geno @ self.beta + inter @ self.omega
        if hasattr(self, 'var'):
            return self.imputePheno(data)
        else:
            print('error: need to fit either effect sizes or var components')
            return 
    
    def imputePheno(self, data):
        Yin = self.Y
        m = data.m
        Kin = 1/m * self.G @ self.G.T
        Kout = 1/m * data.geno @ self.G.T
        var = self.var
        Vin = var[0] * Kin + var[1] * Kin * Kin + var[2] * np.eye(self.G.shape[0])
        Vout = var[0] * Kout + var[1] * Kout * Kout
        return Vout @ np.linalg.inv(Vin) @ Yin