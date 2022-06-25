import numpy as np
import itertools
import torch
from scipy.stats import pearsonr


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
            r = pearsonr(p1_perm, p2.reshape(-1,))[0]
            rsquared = r ** 2
            if rsquared > maxCorr: 
                maxCorr = rsquared
                best_perm = perm

        return maxCorr