import numpy as np
import scipy.stats as sp

def Hotelling(X, Y, alpha):
    nx, d = X.shape
    ny = Y.shape[0]
    mx = np.mean(X, 0)
    my = np.mean(Y, 0)
    mdiff = mx-my
    sx = np.cov(X.T)
    sy = np.cov(Y.T)
    s = (sx/nx) + (sy/ny)
    chi2_stat = np.dot(np.linalg.solve(s, mdiff), mdiff)
    
    pvalue = sp.chi2.sf(chi2_stat, d)
    results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': chi2_stat,
                'H0_rejected': pvalue <= alpha}
    return results