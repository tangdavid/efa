import numpy as np
from scipy import linalg, stats
from sklearn import linear_model
from sklearn import decomposition
import time

class dataset:
    def __init__(self, n, m, k = 1, h2 = 0.9, noise = 0):
        self.n = n 
        self.m = m 
        self.k = k 
        self.h2 = h2
        self.noise = noise

        self.simGeno()
        self.simEffects()
        self.simPheno()

    def simGeno(self):
        geno = np.zeros([self.n, self.m])
        for i in range(self.m):
            p = np.random.beta(2, 2)
            snps = np.random.binomial(2, p, self.n)
            geno.T[i] = snps - (2*p)/np.sqrt(2*p*(1-p))

        inter = linalg.khatri_rao(geno.T, geno.T).T
        self.geno, self.inter = geno, inter

    def simEffects(self):
        u = np.random.normal(0, 1, self.m * self.k).reshape(self.m, -1) 
        v = np.random.normal(0, 1, self.m * self.k).reshape(self.m, -1) 
        noise = np.random.normal(0, self.noise, self.m ** 2).reshape(self.m, -1) 
        noise = np.tril(noise) + np.tril(noise, -1).T
        omega = noise + u @ v.T + v @ u.T 

        self.uv = np.column_stack((u, v)) 
        self.vu = np.column_stack((v, u)) 

        self.u = u 
        self.v = v 
        self.omegaMat = omega
        self.omega = omega.reshape(1, -1)[0]
        
    
    def simPheno(self):
        mean = np.matmul(self.inter, self.omega)
        var = np.var(mean) * (1 - self.h2)/self.h2
        sd = np.sqrt(var)
        noise = np.random.normal(0, sd, self.n)
        self.pheno = mean + noise
        
class decomp:
    def __init__(self):
        self.data = None
    def simData(self, n, m, k = 1, h2 = 0.9, noise = 0):
        self.data = dataset(n, m, k = k, h2 = h2, noise = noise)
    def fitMarginal(self):
        lm = linear_model.LinearRegression(fit_intercept=True)
        omegaHat = list()
        for i in range(self.data.m):
            for j in range(self.data.m):
                interIndex = i * self.data.m + j
                maini = self.data.geno.T[i]
                mainj = self.data.geno.T[j]
                interij = self.data.inter.T[interIndex]
                
                regress = np.vstack((interij, maini, mainj)).T
                
                lm.fit(regress, self.data.pheno)
                omegaHat.append(lm.coef_[0])

        omegaHat = np.array(omegaHat)
        omegaHatMat = omegaHat.reshape(self.data.m, -1)

        self.omegaHat, self.omegaHatMat = omegaHat, omegaHatMat

    def fitRidge(self):
        lm = linear_model.Ridge()
        lm.fit(self.data.inter, self.data.pheno)
        omegaHat = lm.coef_
        omegaHatMat = omegaHat.reshape(self.data.m, -1)

        self.omegaHat, self.omegaHatMat = omegaHat, omegaHatMat



    def fitSVD(self):
        rank = self.data.k * 2
        lm = linear_model.LinearRegression(fit_intercept=True)

        A, singular, B = linalg.svd(self.omegaHatMat)

        A = A[:, :rank]
        singular = singular[:rank]
        B = B[:, :rank]


        lm.fit(self.data.uv, A)
        transform = lm.coef_.T
        
        self.u = (A @ linalg.inv(transform))[:,:self.data.k]
        self.v = (A @ linalg.inv(transform))[:,self.data.k:]

    def symmetricDecomp(self):
        m = self.data.m
        
        thresh = 0.0001
                
        u = np.random.rand(m, 1)
        v = np.random.rand(m, 1)
        
        uprev = np.copy(u)
        vprev = np.copy(v)
        
        i = 0
        
        while(True):
            if i % 2 == 0:
                u = linalg.inv(v @ v.T + (v.T @ v)*np.eye(m)) @ self.omegaHatMat @ v
            else:
                v = linalg.inv(u @ u.T + (u.T @ u)*np.eye(m)) @ self.omegaHatMat @ u
            
            udiff = linalg.norm(u - uprev)
            vdiff = linalg.norm(v - vprev)
            
            if udiff < thresh and vdiff < thresh:
                break
            else:
                uprev = np.copy(u)
                vprev = np.copy(v)
            i += 1

        self.u = u
        self.v = v   
        
    def directDecomp(self):
        m = self.data.m
        
        G = self.data.geno
        Y = self.data.pheno.reshape(-1, 1)
        
        thresh = 0.0001
                
        u = np.random.rand(m, 1)
        v = np.random.rand(m, 1)
        
        uprev = np.copy(u)
        vprev = np.copy(v)
        
        i = 0
        
        while(True):
            if i % 2 == 0:
                A = (G @ v)
                u = 1/2 * linalg.inv((G.T) @ (A * A * G)) @ G.T @ (A * Y)
            else:
                A = (G @ u)
                v = 1/2 * linalg.inv((G.T) @ (A * A * G)) @ G.T @ (A * Y)
            
            udiff = linalg.norm(u - uprev)
            vdiff = linalg.norm(v - vprev)
            
            if udiff < thresh and vdiff < thresh:
                break
            else:
                uprev = np.copy(u)
                vprev = np.copy(v)
            i += 1

        self.u = u
        self.v = v   

    def evalNorm(self):
        
        trueU = self.data.u.reshape(1, -1)[0]
        trueV = self.data.v.reshape(1, -1)[0]
        
        estU = self.u.reshape(1, -1)[0]
        estV = self.v.reshape(1, -1)[0]
        
        original = ("original", 
                    stats.pearsonr(estU, trueU)[0] ** 2,
                    stats.pearsonr(estV, trueV)[0] ** 2)
        switched = ("switched", 
                    stats.pearsonr(estV, trueU)[0] ** 2, 
                    stats.pearsonr(estU, trueV)[0] ** 2)
        
        
        if original[1] + original[2] > switched[1] + switched[2]:
            return original
        else: 
            return switched