from tools import *
from sklearn.decomposition import PCA
                
class CoordinatedModel:
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
            
        
    def getWeights(self, data, pathways, diag = True):
        G = data.geno
        Y = data.pheno
        k = self.k
        
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
            interEffect = linalg.khatri_rao(GU.T, GU.T).T @ weights.reshape(-1, 1)
            mainEffect = np.sum(G @ pathways, axis=1, keepdims=True)
            penalty = reg * np.sum(np.abs(pathways))
            loss = linalg.norm(Y - mainEffect - interEffect)**2 + penalty
        
        return loss
    
    def omegaPCA(self, k=2):
        omega_mat = self.omega.reshape(self.m, self.m)
        pca = PCA(n_components=k)
        pca.fit(omega_mat)
        self.pathways = pca.components_.T

    def evalPathwayAcc(self, data):
        k = self.k
        withAnchors = tools.evalAcc(self.pathways, data.pathways)
        withoutAnchors = tools.evalAcc(self.pathways[:-k], data.pathways[:-k])
        return (withAnchors, withoutAnchors)
    
    def evalBetaAcc(self, data):
        return stats.pearsonr(self.beta.reshape(-1,), data.beta.reshape(-1,))[0] ** 2
    
    def evalOmegaAcc(self, data):
        return stats.pearsonr(self.omega.reshape(-1,), data.omega.reshape(-1,))[0] ** 2
    
    def evalPhenoAcc(self, data):
        prediction = self.predictPheno(data)
        return stats.pearsonr(prediction.reshape(-1,), data.pheno.reshape(-1,))[0] ** 2
    
    def predictPheno(self, data):
        G = data.geno
        U = self.pathways
        weights = self.weights
        
        GU = G @ U
        interEffect = linalg.khatri_rao(GU.T, GU.T).T @ weights.reshape(-1, 1)
        mainEffect = np.sum(GU, axis=1, keepdims=True)
        
        return mainEffect + interEffect
        
    def plotLoss(self):
        plt.plot(self.loss)
        plt.show()
        
class AdditiveModel:
    def fitModel(self, data):
        lr = LinearRegression()
        lr.fit(data.geno, data.pheno)
        self.beta = lr.coef_.reshape(-1, 1)
        
    def predictPheno(self, data):
        return data.geno @ self.beta
    
    def evalBetaAcc(self, data):
        return stats.pearsonr(self.beta.reshape(-1,), data.beta.reshape(-1,))[0] ** 2
    
    def evalPhenoAcc(self, data):
        prediction = self.predictPheno(data)
        return stats.pearsonr(prediction.reshape(-1,), data.pheno.reshape(-1,))[0] ** 2

class UncoordinatedModel:
    def fitBeta(self, data):
        lr = LinearRegression()
        lr.fit(data.geno, data.pheno)
        self.beta = lr.coef_.reshape(-1, 1)
        
    def fitOmega(self, data):
        lr = linear_model.LinearRegression(fit_intercept=True)
        
        Y = data.pheno - data.geno @ self.beta
        inter = linalg.khatri_rao(data.geno.T, data.geno.T).T
        
        omega = np.zeros(data.m ** 2)
        for i in range(data.m):
            for j in range(data.m):
                idx = i * data.m + j    
                interij = inter.T[idx]
                regressors = interij.reshape(-1, 1)
                lr.fit(regressors, Y)
                omega[idx] = lr.coef_.reshape(-1,)[0]
        self.omega = omega.reshape(-1, 1)

    def omegaPCA(self, k=2):
        omega_mat = self.omega.reshape(self.m, self.m)
        pca = PCA(n_components=k)
        pca.fit(omega_mat)
        self.pathways = pca.components_.T
    
    def fitModel(self, data, random_effects = False):
        if random_effects:
            beta, omega, _ = randomEffects.aiML(data)
            self.beta = beta
            self.omega = omega
        else:
            self.fitBeta(data)
            self.fitOmega(data)
        self.m = data.m
        self.n = data.n
        
    def predictPheno(self, data):
        inter = linalg.khatri_rao(data.geno.T, data.geno.T).T
        return data.geno @ self.beta + inter @ self.omega
    
    def evalOmegaAcc(self, data):
        return stats.pearsonr(self.omega.reshape(-1,), data.omega.reshape(-1,))[0] ** 2
    
    def evalBetaAcc(self, data):
        return stats.pearsonr(self.beta.reshape(-1,), data.beta.reshape(-1,))[0] ** 2
    
    def evalPhenoAcc(self, data):
        prediction = self.predictPheno(data)
        return stats.pearsonr(prediction.reshape(-1,), data.pheno.reshape(-1,))[0] ** 2

    def evalPathwayAcc(self, data):
        k = self.pathways.shape[1]
        withAnchors = tools.evalAcc(self.pathways, data.pathways)
        withoutAnchors = tools.evalAcc(self.pathways[:-k], data.pathways[:-k])
        return (withAnchors, withoutAnchors)
    
class Inference:
    def likelihoodRatioTest(data, k):
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
