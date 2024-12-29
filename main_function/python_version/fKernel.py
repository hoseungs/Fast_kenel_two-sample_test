import numpy as np
import scipy.stats as sp
import util as util
import math

#====================================== Testing and the test statistic

def fkernel(X, Y):
    m,d = X.shape
    n,d = Y.shape
    
    b = math.floor( np.sqrt( (m+n)/2 ) )
    
    B1 = math.floor(m/b)
    B2 = math.floor(n/b)
    
    rx = m - B1*b
    ry = n - B2*b
    
    if rx > 0:
        B1i = np.repeat([B1,B1+1],[b-rx,rx])
    else:
        B1i = np.repeat(B1,b)
        
    if ry > 0:
        B2i = np.repeat([B2,B2+1],[b-ry,ry])
    else:
        B2i = np.repeat(B2,b)
    
    Bi = B1i + B2i
    
    p1i = B1i*(B1i-1)/Bi/(Bi-1)
    p2i = p1i*(B1i-2)/(Bi-2)
    p3i = p2i*(B1i-3)/(Bi-3)

    q1i = B2i*(B2i-1)/Bi/(Bi-1)
    q2i = q1i*(B2i-2)/(Bi-2)
    q3i = q2i*(B2i-3)/(Bi-3)
  
    list_Z_w = np.zeros(b)
    list_Z_d = np.zeros(b)
    u1 = v1 = u2 = v2 = 0
    
    for j in range(b):
        u1 = v1 
        v1 = u1 + B1i[j] 
        u2 = v2
        v2 = u2 + B2i[j] 
        Xi = X[u1:v1,:]
        Yi = Y[u2:v2,:]
        
        Z_wi, Z_di = compute_stat_FGker(Xi, Yi, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
        
        list_Z_w[j] = Z_wi
        list_Z_d[j] = Z_di
    
    Zwbar = np.mean(list_Z_w)
    Zdbar = np.mean(list_Z_d)
    
    pw = sp.norm.sf(np.sqrt(b)*Zwbar)
    pd = 2*sp.norm.sf(np.sqrt(b)*abs(Zdbar))
    
    temp = sorted([pw, pd])
    pvalue = 2*min(temp[0], temp[1])
    
    return pvalue



def compute_stat_FGker(X, Y, B1, B2, B, p1, p2, p3, q1, q2, q3):
    Z = np.vstack((X,Y))
    Dis = util.dist_matrix(Z, Z)
    
    # compute the median heuristic
    Itri = np.tril_indices(Dis.shape[0], -1)
    Tri = Dis[Itri]
    med = np.median(Tri)
    if med <= 0:
        # use the mean
        gwidth = np.mean(Tri)
    else:
        gwidth = med
    
    K = np.exp( -Dis**2 / (2*gwidth**2) ) - np.eye(B)
    
    R0 = np.sum(K)
    R1 = np.sum(K**2)
    R2 = np.sum(np.sum(K,1)**2) - R1
    R3 = R0**2 - 2*R1 - 4*R2
    
    Kx = np.sum(K[0:B1, 0:B1])/B1/(B1-1)
    Ky = np.sum(K[B1:B, B1:B])/B2/(B2-1)
    
    mu = R0/B/(B-1)
    varx = (2*R1*p1 + 4*R2*p2 + R3*p3)/B1/B1/(B1-1)/(B1-1) - mu**2
    vary = (2*R1*q1 + 4*R2*q2 + R3*q3)/B2/B2/(B2-1)/(B2-1) - mu**2
    cova = R3/B/(B-1)/(B-2)/(B-3) - mu**2
    
    # test statistic Z_d
    u_d = B1*(B1-1)
    v_d = -B2*(B2-1)
    mean_d = mu*u_d + mu*v_d
    var_d = (u_d**2)*varx + (v_d**2)*vary + 2*u_d*v_d*cova
    Z_d = (Kx*u_d + Ky*v_d - mean_d)/np.sqrt(var_d)
    
    # test statistic Z_w
    u_w = B1/B
    v_w = B2/B
    mean_w = mu*u_w + mu*v_w
    var_w = varx*u_w**2 + vary*v_w**2 + 2*u_w*v_w*cova
    Z_w = (Kx*u_w + Ky*v_w - mean_w)/np.sqrt(var_w)
    
    return (Z_w, Z_d)
