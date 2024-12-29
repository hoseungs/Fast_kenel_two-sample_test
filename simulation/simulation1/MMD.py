import numpy as np
import scipy.stats as sp
import util as util
import math

def MMDtest(X, Y, gwidth, bsize, alpha):
    # gwidth : gamma. The square root of bandwidth.
    # bsize : box size.
    
    n,d = X.shape
    
    test_stat, vari = compute_stat_MMD(X, Y, gwidth, bsize)
    
    pvalue = sp.norm.sf(test_stat, loc=0, scale=(bsize*vari/n)**0.5)
    
    results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                'H0_rejected': pvalue <= alpha}
    return results


def compute_stat_MMD(X, Y, gwidth, bsize):
    n,d = X.shape
    
    bsize = math.floor(bsize)
    m = math.floor(n/bsize)
    
    list_stat = np.zeros(m)
    
    for i in range(m):
        Xi = X[range(i*bsize,((i+1)*bsize)),:]
        Yi = Y[range(i*bsize,((i+1)*bsize)),:]
        
        stat = compute_MMD(Xi, Yi, gwidth, bsize)
        
        list_stat[i] = stat
    
    test_stat = np.mean(list_stat)
    vari = np.var(list_stat)
    
    return (test_stat, vari)

def compute_MMD(X, Y, gwidth, bsize):
    Z = np.vstack((X,Y))
    Dis2 = util.dist_matrix(Z, Z)
    K = np.exp( -Dis2**2 / (2.0 * (gwidth**2)) ) - np.eye(2*bsize)
    
    Kx = np.sum(K[0:bsize, 0:bsize])/bsize/(bsize-1)
    Ky = np.sum(K[bsize:, bsize:])/bsize/(bsize-1)
    Kxy = np.sum(K[0:bsize, bsize:])/bsize/bsize
    
    stat = Kx + Ky - 2*Kxy
    
    return stat

def MMD_grid_search_gwidth(X, Y, bsize, alpha):
    n,d = X.shape
    
    med = util.meddistance(np.vstack((X,Y)), 1000)
    list_gwidth_temp = np.hstack( ( (med**2) *(2.0**np.linspace(-3, 4, 30) ) ) )
    list_gwidth_temp.sort()
    
    powers = np.zeros(len(list_gwidth_temp))
    
    for wi, gwidth in enumerate(list_gwidth_temp):
        stat, vari = compute_stat_MMD(X, Y, np.sqrt(gwidth), bsize)
        thresh = sp.norm.isf(alpha, loc=0, scale=(bsize*vari/n)**0.5)
        power = sp.norm.sf(thresh, loc=stat, scale=(bsize*vari/n)**0.5)
        powers[wi] = power
        
    besti = np.argmax(powers)
    gwidth0 = list_gwidth_temp[besti]
    gwidth00 = np.sqrt(gwidth0)
    return gwidth00