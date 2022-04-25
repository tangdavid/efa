import pandas as pd
from sklearn.model_selection import train_test_split
from tools import *

class SimDataset:
    def __init__(self, n, m, k = 2,
                 h2 = 0.7, noise_beta = 0, noise_omega = 0, additive_model_var = 0.5,
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
        #var_eomega = np.var(omega) * (self.noise_omega/ (1 - self.noise_omega)) 
        var_eomega = np.var(omega) * (self.noise_omega) 
        eomega = np.random.normal(0, 1, size = (m, m))
        eomega *= np.sqrt(var_eomega) / np.std(eomega)
        eomega = np.tril(eomega) + np.tril(eomega, -1).T
        weights *= np.sqrt((1 - self.noise_omega))
        
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


class RealDataset:
    def __init__(self, *args, **kwargs):
        infile = kwargs.get('infile', None)
        if infile:
            df = pd.read_table(infile, sep = ' ')
            df = df[df['PHENOTYPE'] != -9]
            self.geno = df.iloc[:, 6:].to_numpy()
            self.pheno = df.iloc[:, 5].to_numpy().reshape(-1, 1)
            self.n, self.m = self.geno.shape
        else:
            self.geno = kwargs.get('geno')
            self.pheno = kwargs.get('pheno')
            self.n, self.m = self.geno.shape

def splitTrain(data):
    train_G, test_G, train_Y, test_Y = train_test_split(data.geno, data.pheno, test_size=0.2)
    train = RealDataset(geno = train_G, pheno = train_Y)
    test = RealDataset(geno = test_G, pheno = test_Y)
    return(train, test)

def splitKFold(data, folds = 10):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds, shuffle=True)
    kf.split(data.geno)
    
    res = list()
    for train_idx, test_idx in kf.split(data.geno):
        train = RealDataset(geno = data.geno[train_idx,], pheno = data.pheno[train_idx,])
        test = RealDataset(geno = data.geno[test_idx,], pheno = data.pheno[test_idx,])
        res.append((train, test))
    return res