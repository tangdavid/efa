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
    def __init__(self, k, sink = False):
        self.k = k if not sink else k + 1
        self.sink = sink
        self.loss = None
        self.conv = True
        self.pathways = None
        self.weights = None

        weights_mask = np.ones((self.k, self.k))
        if sink:
            weights_mask[:,-1] = 0
            weights_mask[-1,:] = 0
        self.weights_mask = weights_mask


    def gradDescent(self, data, args):
        additive_init = args.get('additive_init', True)
        init_noise = args.get('init_noise', 0.1)
        self_interactions = args.get('self_interactions', False)
        anchors = args.get('anchors', None)
        progress = args.get('progress', False)

        m = data.m
        k = self.k

        tol = 1e-6

        # for anchor snps
        pathways_mask = torch.ones(size = (m, k))
        if anchors:
            pathways_mask[-k:, :] = torch.eye(k)

        weights, pathways = self.initPathways(data, additive_init, init_noise)
        weights = torch.tensor(weights, requires_grad = True)
        pathways = torch.tensor(pathways, requires_grad = True)
        
        # use Adam to optimize 
        params = [weights, pathways]
        optimizer = optim.Adam(params)

        prevLoss = currentLoss = self.getLoss(data, pathways, weights).item()
        lossList = [currentLoss]
        iterations = 0

        conv = True
        
        weights_mask = torch.ones(size = (k, k))
        if not self_interactions: weights_mask = weights_mask - torch.eye(k)
        
        while True:            
            iterations += 1
            
            loss = self.getLoss(
                data, 
                pathways*pathways_mask, 
                weights*weights_mask
            )
            
            # optimize the loss function with gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # monitor convergence
            prevLoss = currentLoss
            currentLoss = loss.item()
            lossList.append(currentLoss)
            
            if np.abs(currentLoss - prevLoss)/prevLoss < tol and iterations > 1:
                 break
            if iterations % 1000 == 0 and progress: 
                print("(iterations,loss):", iterations, round(loss.item(), 3), flush=True)
                
            if iterations > 15000: 
                print("warning: failed to converge", flush=True)
                conv = False
                break

        pathways = pathways.detach().numpy()
        weights = weights.detach().numpy()
        
        res = dict()
        res['loss'] = lossList
        res['pathways'] = pathways
        res['beta'] = np.sum(pathways, axis = 1, keepdims=True)
        res['weights'] = weights
        res['omega'] = tools.outer(pathways, weights).reshape(-1, 1)
        res['conv'] = conv
        return res

    def coordDescent(self, data, args):
        G = data.geno
        Y = data.pheno
        k = self.k
        
        tol = 1e-6
        additive_init = args.get('additive_init', True)
        init_noise = args.get('init_noise', 0.1)
        progress = args.get('progress', False)
                
        # initialize pathways and weights
        weights, pathways = self.initPathways(data, additive_init, init_noise)    

        iterations = 0
        currentLoss = self.getLoss(data, pathways, weights, tensor = False)
        prevLoss = currentLoss
        lossList = [currentLoss]
        conv = True
        
        while(True):
            # iteratively update each pathway
            for i in range(k):
                # compute constants
                pathway_mask = [True] * k
                pathway_mask[i] = False
                weights_mask = np.ones_like(weights)
                weights_mask[i,:] = 0
                weights_mask[:,i] = 0
                weights_mask *= self.weights_mask
                np.fill_diagonal(weights_mask, 0)

                X = khatri_rao((G @ pathways).T, (G @ pathways).T).T
                C1 = X @ (weights * weights_mask).reshape(-1, 1)
                C2 = G @ pathways[:,pathway_mask].sum(axis=1, keepdims=True) 
                C = C1 + C2

                A = G @ pathways @ weights[i].reshape(-1, 1)
                
                # equations from the first order conditions
                B1 = (4*G.T @ (A*A*G)) + (4*G.T @ (A*G)) + (G.T @ G)
                B2 = (2*(A*G).T @ Y) + (G.T @ Y) - (2*(A*G).T @ C) - (G.T @ C)
                ui = np.linalg.solve(B1, B2)
                pathways[:, i] = ui.reshape(-1,)
                
                
            # iteratively fit weights
            for i in range(k):
                if i == k-1 and self.sink: continue
                for j in range(i):
                    
                    ui = pathways[:,i]
                    uj = pathways[:,j]

                    weights_mask = np.ones_like(weights)
                    weights_mask[i,j] = 0
                    weights_mask[j,i] = 0
                    np.fill_diagonal(weights_mask, 0)
                    weights_mask *= self.weights_mask
                    
                    # compute constants
                    pair = ((G @ ui)*(G @ uj)).reshape(-1, 1)
                    X = khatri_rao((G @ pathways).T, (G @ pathways).T).T
                    C1 = X @ (weights * weights_mask).reshape(-1, 1)
                    C2 = G @ pathways.sum(axis=1, keepdims=True) 
                    C = C1 + C2
                    
                    # equation from the first order conditions
                    weight = 1/2 * (pair.T @ (Y - C))/(pair.T @ pair)
                    weights[i, j] = weight
                    weights[j, i] = weight    

            # zero out sink pathway if we are fitting it
            weights *= self.weights_mask
            # monitor convergence
            iterations += 1
            prevLoss = currentLoss
            currentLoss = self.getLoss(data, pathways, weights, tensor = False)
            lossList.append(currentLoss)
            
            if np.abs(currentLoss - prevLoss)/prevLoss < tol and iterations > 1:
                 break

            if iterations % 100 == 0 and progress: 
                print("(iterations,loss):", iterations, round(currentLoss, 3), flush=True)

            if iterations > 1500:
                print("warning: failed to converge", flush=True)
                conv = False
                break

        res = dict()    
        res['loss'] = lossList
        res['pathways'] = pathways
        res['beta'] = np.sum(pathways, axis = 1, keepdims=True)
        res['weights'] = weights
        res['omega'] = tools.outer(pathways, weights).reshape(-1, 1)
        res['conv'] = conv
        return res

    def initPathways(
        self, 
        data, 
        additive_init,
        init_noise = 0.1,
    ):
        m = data.m
        k = self.k

        pathways = np.zeros((m, k))
        const = 1
        if additive_init:
            lr = LinearRegression()
            lr.fit(data.geno, data.pheno)
            additive = lr.coef_.reshape(-1,)
            for i in range(k):
                pathways[:, i] = additive/k
            const = np.abs(pathways.sum(axis = 1)).mean()
        
        init_noise *= const 
        weights = np.random.normal(0, init_noise, size = (k, k))
        weights = np.tril(weights, -1) + np.tril(weights, -1).T    
        pathways += np.random.normal(0, init_noise, size = (m, k))
        return weights, pathways
        
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

    def getLoss(self, data, pathways, weights, tensor = True):
        G = torch.tensor(data.geno).double() if tensor else data.geno
        Y = torch.tensor(data.pheno).double() if tensor else data.pheno

        if tensor:
            GU = G @ pathways
            interEffect = tools.face_splitting(GU, GU) @ weights.view(-1, 1)
            mainEffect = torch.sum(G @ pathways, dim=1, keepdims=True)
            loss = torch.norm(Y - mainEffect - interEffect)**2
        else:
            GU = G @ pathways
            interEffect = khatri_rao(GU.T, GU.T).T @ weights.reshape(-1, 1)
            mainEffect = np.sum(G @ pathways, axis=1, keepdims=True)
            loss = norm(Y - mainEffect - interEffect)**2
        
        return loss

    def plotLoss(self):
        plt.plot(self.loss)
        plt.show()

    def fitModel(self, data, **kwargs):

        restarts = kwargs.get('restarts', 10)
        algo = kwargs.get('algo', 'coord')

        minLoss = float('inf')
        minLossRes = None

        for restart in range(restarts):
            print(f'restart {restart + 1}', flush=True)
            if algo == 'grad':
                res = self.gradDescent(data, kwargs)
            elif algo == 'coord':
                res = self.coordDescent(data, kwargs)
            else:
                print('error: algorithm must be either grad or coord')
                return -1

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
