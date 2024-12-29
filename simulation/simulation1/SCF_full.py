"""
import SCF_full as sdf

test_locs, gwidth = scf.SCFopt(X_tr, Y_tr, J)
res = scf.SCFtest(X_te, Y_te, test_locs, gwidth, alpha)
print(res)
"""

import numpy as np
import scipy.stats as sp
import util as util
import torch

#====================================== Testing and the test statistic

def SCFtest(X, Y, test_locs, gwidth, alpha):
    """ 
    Two-sample test using squared difference of mean embedding functions, evaluated 
    at a finite set of test locations. Use Gaussian kernel.
    ----------
    X : Samples from distribution P (array), shape = [# of samples (n), dimension (d)]
    Y : Samples from distribution Q (array), shape = [n, d]
    test_locs : Finite set of test locations (array), shape = [# of locations (J), d]
    gwidth : The square Gaussian width (float)
    alpha : The level of the test (float)
    -------
    results : dictionary
    {alpha: float, pvalue: float, , test statistic: float, h0_rejected: Boolean}
    """
    J, d = test_locs.shape
    S = compute_stat_SCF(X, Y, test_locs, gwidth) # test statistic
    pval = sp.chi2.sf(S, 2*J) # 2J degrees of freedom because of sin and cos
    results = {'alpha': alpha, 'p-value': pval, 'test statistic': S, 'H0_rejected': pval<=alpha}
    return results

def compute_stat_SCF(X, Y, test_locs, gwidth):
    """ 
    Two-sample test statistic using squared difference of mean embedding functions
    , evaluated at a finite set of test locations. Use Gaussian kernel.
    ----------
    X : Samples from distribution P (array), shape = [# of samples (n), dimension (d)]
    Y : Samples from distribution Q (array), shape = [n, d]
    test_locs : Finite set of test locations (array), shape = [# of locations (J), d]
    gwidth : The square Gaussian width (float)
    -------
    results : The test statistic S (float)
    """
    if test_locs is None:
        raise ValueError('test_locs must be specified')
    if gwidth is None or gwidth <=0:
        raise ValueError('require gaussian width > 0')
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Sample size n must be the same for X and Y.')
        
    n, d = X.shape
    J = test_locs.shape[0]
    
    X = X/np.sqrt(gwidth)
    Y = Y/np.sqrt(gwidth)
        
    # inverse Fourier transform (upto scaling) of the unit-width Gaussian kernel 
    fx = np.exp(-np.sum(X**2, 1)/2).reshape(n, 1)
    fy = np.exp(-np.sum(Y**2, 1)/2).reshape(n, 1)
        
    # n x J
    x_freq = np.dot(X, test_locs.T).reshape(n, J)
    y_freq = np.dot(Y, test_locs.T).reshape(n, J)
        
    # zx: n x 2J
    zx = np.hstack((np.sin(x_freq)*fx, np.cos(x_freq)*fx))
    zy = np.hstack((np.sin(y_freq)*fy, np.cos(y_freq)*fy))
    Z = zx-zy
    
    assert Z.shape == (n, 2*J)
    
    Sig = np.cov(Z.T)
    W = np.mean(Z, 0)
    
    reg = 1e-5
    S = n*np.linalg.solve(Sig + reg*np.eye(2*J), W).dot(W)
    
    return S



#============================================== Parameter optimization
    
def SCF_grid_search_gwidth(X, Y, T, list_gwidth, alpha):
    """
    Linear search for the best Gaussian width in the list that maximizes 
    the test power, fixing the test locations to T. 
    The test power is given by the CDF of a non-central Chi-squared distribution.
    ----------
    X : Samples from distribution P (array), shape = [n, d]
    Y : Samples from distribution Q (array), shape = [n, d]
    T : Initial test locations (array), shape = [J, d]
    list_gwidth: Candidates of the Gaussian width (array)
    alpha : The level of the test (float)
    -------
    results : Best Guassian width index
    """
    
    J = T.shape[0]
    
    powers = np.zeros(len(list_gwidth))
    lambs = np.zeros(len(list_gwidth))
    thresh = sp.chi2.isf(alpha, df=2*J)
    
    for wi, gwidth in enumerate(list_gwidth):
        # non-centrality parameter
        try:
            lamb = compute_stat_SCF(X, Y, T, gwidth)
            if lamb <= 0:
                # This can happen when Z, Sig are ill-conditioned. 
                raise np.linalg.LinAlgError
            if np.iscomplex(lamb):
                # complext value can happen if the covariance is ill-conditioned?
                print('Lambda is complex. Truncate the imag part. lamb: %s'%(str(lamb)))
                lamb = np.real(lamb)

            power = sp.ncx2.sf(thresh, df=2*J, nc=lamb)
            powers[wi] = power
            lambs[wi] = lamb
        
        except np.linalg.LinAlgError:
            # probably matrix inverse failed. 
            print('LinAlgError. skip width (%d, %.3g)'%(wi, gwidth))
            powers[wi] = np.NINF
            lambs[wi] = np.NINF
    # to prevent the gain of test power from numerical instability, 
    # consider upto 3 decimal places. Widths that come early in the list 
    # are preferred if test powers are equal.
    besti = np.argmax(np.around(powers, 3))
    return besti

def SCFopt(X, Y, alpha, n_test_locs=10, max_iter=400, 
            locs_step_size=0.1, gwidth_step_size=0.01, batch_proportion=1.0, 
            tol_fun=1e-3):
        """
        Optimize the test locations and the Gaussian kernel width by 
        maximizing the test power. X, Y should not be the same data as used 
        in the actual test (i.e., should be a held-out set). 
        ----------
        max_iter : gradient descent iterations
        batch_proportion : (0,1] value to be multipled with nx giving the batch 
            size in stochastic gradient. 1 = full gradient ascent.
        tol_fun : termination tolerance of the objective value
        ----------
        results : (test_locs, gaussian_width) (tuple)
        """
        
        n, d = X.shape
        J = n_test_locs

        # Fit a Gaussian to each dataset and draw half of n_test_locs from each. 
        T0 = util.init_locs_randn2(X, Y, J)
        
        # Use grid search to initialize the gwidth
        med = util.meddistance(np.vstack((X,Y)), 1000)
        list_gwidth_temp = np.hstack( ( (med**2) *(2.0**np.linspace(-3, 4, 30) ) ) )
        list_gwidth_temp.sort()
        
        besti = SCF_grid_search_gwidth(X, Y, T0, list_gwidth_temp, alpha)
        gwidth0 = list_gwidth_temp[besti]
        
        assert util.is_real_num(gwidth0), 'gwidth0 not real. Was %s'%str(gwidth0)
        assert gwidth0 > 0, 'gwidth0 not positive. Was %.3g'%gwidth0
        
        param = np.ones((J, d))
        param[:, :] = T0
        param = np.ravel(param, order="C")
        param = np.hstack((param, np.array([np.sqrt(gwidth0)])))
        
        def gain_torch(param):
            """ Compute f using pytorch expressions """
            if not (isinstance(param, torch.Tensor)):
                param = torch.tensor(param, dtype=torch.float64, requires_grad=True)

            X_new = torch.from_numpy(X)
            Y_new = torch.from_numpy(Y)

            X_new = X_new / param[J * d]
            Y_new = Y_new / param[J * d]
            
            fx = torch.exp((-1 / 2) * torch.sum(X_new ** 2, 1)).reshape(n, 1)
            fy = torch.exp((-1 / 2) * torch.sum(Y_new ** 2, 1)).reshape(n, 1)
            
            x_freq = torch.matmul(
                X_new, torch.reshape(param[: J * d], (J, d)).t()
            ).reshape(n, J)
            y_freq = torch.matmul(
                Y_new, torch.reshape(param[: J * d], (J, d)).t()
            ).reshape(n, J)

            z_x = torch.cat((torch.sin(x_freq)*fx, torch.cos(x_freq)*fx), 1)
            z_y = torch.cat((torch.sin(y_freq)*fy, torch.cos(y_freq)*fy), 1)
            
            z_0 = z_x - z_y
            W0 = torch.mean(z_0, 0)
            
            cov0 = z_0 - torch.mean(z_0, 0)
            cov0 = (1 / (cov0.shape[0] - 1)) * torch.matmul(torch.t(cov0), cov0)

            reg = 1e-5
            cov0 = cov0 + reg * torch.eye(2*J).double()
            u, D, v = torch.svd(cov0, some=True)
            D = torch.diag(D)
            square_root = torch.matmul(torch.matmul(u, torch.sqrt(D)), v.t())

            S = (torch.Tensor([n]).double()) * (
                torch.matmul(W0.view(1,2*J), torch.inverse(square_root)).matmul(W0)
            )
            
            return S

        def grad_gain_torch(param):
            """ Compute the gradient of f using pytorch's autograd """
            param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
            stat = gain_torch(param)
            stat.backward()
            return param.grad.numpy()

        step_pow = 0.5
        max_gam_sq_step = 1.0
        
        # gradient_ascent
        for t in range(max_iter):
            gain_old = gain_torch(param)
            gradient = grad_gain_torch(param)

            grad_T = gradient[: J * d]
            grad_gwidth = gradient[J * d]

            param[: J * d] = (
                param[: J * d]
                + locs_step_size
                * grad_T
                / (t + 1) ** step_pow
                / np.sum(grad_T ** 2) ** step_pow
            )

            update_gwidth = (
                gwidth_step_size
                * np.sign(grad_gwidth)
                * min(np.abs(grad_gwidth), max_gam_sq_step)
                / (t + 1) ** step_pow
            )
            param[J * d] = param[J * d] + update_gwidth

            if param[J * d] < 0:
                param[J * d] = np.abs(param[J * d])

            if t >= 2 and abs(gain_torch(param) - gain_old) <= tol_fun:
                return (param[: J * d].reshape(J, d), param[J * d] ** 2)

        T = param[: J * d].reshape(J, d)
        gwidth2 = param[J * d] ** 2 # The square Gaussian width

        return (T, gwidth2)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    












