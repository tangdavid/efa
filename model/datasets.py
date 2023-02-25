import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from scipy.linalg import khatri_rao, norm
from scipy import stats
from tools import tools

class SimDataset:
    def __init__(
        self, n, m, 
        k = 2, 
        h2 = 0.7, 
        noise_beta = 0, 
        noise_omega = 0, 
        additive_model_var = 0.5,
        sparse = 0, 
        self_interactions = True, 
        anchor_strength = 0.5, 
        dominance = False,
        store_inter = True
        ):

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
        self.dominance = dominance

        if self.additive_model_var == 0: self.additive_model_var = 1e-7
        
        self.simGeno()
        self.simEffects()
        self.simPheno()
        if not store_inter: del self.inter

    def simGeno(self):
        geno = np.zeros([self.n, self.m])
        for i in range(self.m):
            p = np.random.beta(2, 2)
            p = max(p, 0.05)
            p = min(p, 0.95)
            snps = np.random.binomial(2, p, self.n)
            geno.T[i] = (snps - (2*p))/np.sqrt(2*p*(1-p))

        geno -= np.mean(geno, axis = 0)
        geno /= np.std(geno, axis = 0)

        # interaction effects as khatri rao
        inter = khatri_rao(geno.T, geno.T).T
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
        
        norm_anchors = norm(pathways[-k:, :])
        norm_others = norm(pathways[:-k, :])

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
        var_eomega = np.var(omega) * (self.noise_omega) 
        if self.dominance:
            eomega = np.diag(np.random.normal(0, 1, size = m))
        else:
            eomega = np.random.normal(0, 1, size = (m, m))

        eomega *= np.sqrt(var_eomega) / np.std(eomega)
        eomega = np.tril(eomega) + np.tril(eomega, -1).T
        weights *= np.sqrt((1 - self.noise_omega))
        omega = tools.outer(pathways, weights)
        
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
        self.eomega = eomega.reshape(-1, 1)
        self.weights = weights
        self.normalizePheno()
        
    def normalizePheno(self):
        std_mean = np.std(self.inter @ self.omega + self.geno @ self.beta)
        scale = std_mean / np.sqrt(self.h2)
        self.pathways /= scale
        self.weights *= scale
        self.beta /= scale
        self.omegaMat /= scale
        self.eomega /= scale
        
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

    def permute(self, residualize=None):
        if residualize is None: residualize = np.zeros((self.n, 1))
        residuals = self.pheno - residualize
        residuals = np.random.permutation(residuals)
        self.pheno = residuals + residualize

class SimDatasetLD:
    def __init__(self, n, m, r2 = 0, h2 = 0.5):
        self.inter = None
        self.n = n
        self.m = m
        self.r2 = r2
        self.h2 = h2
        self.simEffects()

    def simEffects(self):
        r = np.sqrt(self.r2)
        cov = np.array(
            [[1, r, r],
             [r, 1, 0],
             [r, 0, 1]]
        )
        geno = np.random.multivariate_normal(
            mean = np.zeros(3),
            cov=cov, 
            size = (self.n, self.m//2)
        )
        geno = np.hstack(np.swapaxes(geno, 0, 1))
        causal = geno[:, ::3]
        geno = np.delete(geno, range(0, self.m + self.m // 2, 3), axis = 1)
        beta = np.random.normal(size = (self.m // 2, 1))
        additive_effect = causal @ beta
        beta *= np.sqrt(self.h2)/np.std(additive_effect)
        additive_effect = causal @ beta

        epsilon = np.random.normal(size = (self.n, 1))
        epsilon *= np.sqrt(1 - self.h2)/ np.std(epsilon)

        self.causal = causal
        self.beta = beta
        self.pheno = additive_effect + epsilon
        self.geno = geno

    def withEffectSizes(self, data):
        self.beta = data.beta
        eps_std = np.sqrt(1 - self.h2)
        epsilon = np.random.normal(0, eps_std, self.n).reshape(-1, 1)
        self.pheno = self.causal @ self.beta + epsilon

    def permute(self, residualize=None):
        if residualize is None: residualize = np.zeros((self.n, 1))
        residuals = self.pheno - residualize
        residuals = np.random.permutation(residuals)
        self.pheno = residuals + residualize

class SimDatasetAdditive:
    def __init__(self, n, m, h2 = 0.5):
        self.inter = None
        self.n = n
        self.m = m
        self.h2 = h2
        self.simGeno()
        self.simEffects()
        self.simPheno()
        pass

    def simGeno(self):
        geno = np.zeros([self.n, self.m])
        for i in range(self.m):
            p = np.random.beta(2, 2)
            p = max(p, 0.05)
            p = min(p, 0.95)
            snps = np.random.binomial(2, p, self.n)
            geno.T[i] = (snps - (2*p))/np.sqrt(2*p*(1-p))

        geno -= np.mean(geno, axis = 0)
        geno /= np.std(geno, axis = 0)
        self.geno = geno

    def simEffects(self):
        self.beta = np.random.normal(0, 1, (self.m, 1))

    def simPheno(self):
        std_mean = np.std(self.geno @ self.beta)
        scale = std_mean / np.sqrt(self.h2)
        self.beta /= scale
        eps_std = np.sqrt(1 - self.h2)
        noise = np.random.normal(0, eps_std, self.n).reshape(-1, 1)
        self.pheno = self.geno @ self.beta + noise

    def permute(self, residualize=None):
        if residualize is None: residualize = np.zeros((self.n, 1))
        residuals = self.pheno - residualize
        residuals = np.random.permutation(residuals)
        self.pheno = residuals + residualize

    def withEffectSizes(self, data):
        self.beta = data.beta
        eps_std = np.sqrt(1 - self.h2)
        noise = np.random.normal(0, eps_std, self.n).reshape(-1, 1)
        self.pheno = self.geno @ self.beta + noise

class RealDataset:
    def __init__(self, rint = False, *args, **kwargs):
        infile = kwargs.get('infile', None)
        delim = kwargs.get('delim', '\t')
        if infile:
            df = pd.read_table(infile, sep = delim)
            df = df[df['PHENOTYPE'] != -9]
            geno = self.standardize(df.iloc[:, 6:].to_numpy())
            pheno = self.standardize(df.iloc[:, 5].to_numpy().reshape(-1, 1))
            n, m = geno.shape
        else:
            geno = self.standardize(kwargs.get('geno'))
            pheno = self.standardize(kwargs.get('pheno'))
            n, m = geno.shape
        if rint: pheno = self.rint(pheno)

        self.geno = geno
        self.pheno = pheno
        self.n = n
        self.m = m
            
    def standardize(self, arr):
        return (arr - arr.mean(axis = 0))/arr.std(axis = 0)

    def rint(self, arr):
        order = arr.argsort(axis=0)
        ranks = order.argsort(axis = 0)
        arr_rint = stats.norm.ppf((ranks+0.5)/arr.shape[0])
        return(arr_rint)
    
    def permute(self, residualize=None, seed=None):
        if residualize is None: residualize = np.zeros((self.n, 1))
        residuals = self.pheno - residualize
        residuals = np.random.RandomState(seed=seed).permutation(residuals)
        self.pheno = residuals + residualize

    def bootstrap(self):
        bootstrap_idx = np.random.choice(np.arange(self.n), self.n)
        geno = self.geno[bootstrap_idx]
        pheno = self.pheno[bootstrap_idx]
        return RealDataset(geno = geno, pheno = pheno)

def splitTrain(data, test_size = 0.2):
    res = train_test_split(data.geno, data.pheno, test_size=test_size)
    train_G, test_G, train_Y, test_Y = res
    train = RealDataset(geno = train_G, pheno = train_Y)
    test = RealDataset(geno = test_G, pheno = test_Y)
    return(train, test)

def splitKFold(data, folds = 10, seed=None):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    kf.split(data.geno)
    
    res = list()
    for train_idx, test_idx in kf.split(data.geno):
        train = RealDataset(
            geno = data.geno[train_idx,], 
            pheno = data.pheno[train_idx,]
        )
        test = RealDataset(
            geno = data.geno[test_idx,], 
            pheno = data.pheno[test_idx,]
        )
        res.append((train, test))
    return res

def generateOOS(data, n):
    if isinstance(data, SimDataset):
        m, k, h2 = data.m, data.k, data.h2
        res = SimDataset(n, m, k = k, h2 = h2)
        res.withEffectSizes(data)
    elif isinstance(data, SimDatasetAdditive):
        m, h2 = data.m, data.h2
        res = SimDatasetAdditive(n, m, h2 = h2)
        res.withEffectSizes(data)

    elif isinstance(data, SimDatasetLD):
        m, r2, h2 = data.m, data.r2, data.h2
        res = SimDatasetLD(n, m, r2 = r2, h2 = h2)
        res.withEffectSizes(data)

    return res    

def concatDatasets(data1, data2):
    G = np.vstack((data1.geno, data2.geno))
    Y = np.vstack((data1.pheno, data2.pheno))
    return RealDataset(geno = G, pheno = Y)
