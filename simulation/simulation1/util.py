import numpy as np


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances of points in the matrix.
    Useful as a heuristic for setting Gaussian kernel's width.
    ----------
    X : data (array), shape = [# of samples (n), dimension (d)]
    subsample : Subsample the samples from X (float)
    mean_on_fail : boolean. If True, use the mean when the median distance is 0. Return
    ------
    results: median distance = square root of gamma (float)
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med
    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size n x n.
    ----------
    X : Samples from distribution P (array), shape = [n, d]
    Y : Samples from distribution Q (array), shape = [n, d]
    -------
    results : distance matrix (array), shape = [2n, 2n]
    """
    sx = np.sum(X**2, 1)
    sy = np.sum(Y**2, 1)
    D2 =  sx[:, np.newaxis] - 2.0*np.dot(X, Y.T) + sy[np.newaxis, :] 
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D

def is_real_num(x):
    """
    Return true if x is a real number
    """
    try:
        float(x)
        return not (np.isnan(x) or np.isinf(x))
    except ValueError:
        return False


def init_locs_randn(X, Y, seed=1):
    """
    Fit a Gaussian to the merged data of the two samples and draw
    1 points from this merged gaussian data
    ----------
    X : Samples from distribution P (array), shape = [n, d]
    Y : Samples from distribution Q (array), shape = [n, d]
    -------
    results: T0 (array), shape = [1, d]
    """

    # set the seed
    rand_state = np.random.get_state()
    np.random.seed(seed)

    d = X.shape[1]
    # fit a Gaussian in the middle of X, Y and draw sample to initialize T
    xy = np.vstack((X, Y))
    mean_xy = np.mean(xy, 0)
    cov_xy = np.cov(xy.T)
    [Dxy, Vxy] = np.linalg.eig(cov_xy + 1e-3 * np.eye(d))
    Dxy = np.real(Dxy)
    Vxy = np.real(Vxy)
    Dxy[Dxy <= 0] = 1e-3
    eig_pow = 0.9  # 1.0 = not shrink
    reduced_cov_xy = Vxy.dot(np.diag(Dxy ** eig_pow)).dot(Vxy.T) + 1e-3 * np.eye(d)

    T0 = np.random.multivariate_normal(mean_xy, reduced_cov_xy, 1)
    # reset the seed back to the original
    np.random.set_state(rand_state)
    return T0
    
def init_locs_randn2(X, Y, n_test_locs):
    """
    Fit a Gaussian to each dataset and draw half of n_test_locs from each. 
    ----------
    X : Samples from distribution P (array), shape = [n, d]
    Y : Samples from distribution Q (array), shape = [n, d]
    n_test_locs : Number of test locations (float)
    -------
    results: T0 (array), shape = [J, d]
    """

    n, d = X.shape

    if n_test_locs == 1:
        T0 = init_locs_randn(X, Y)

    else:
        # fit a Gaussian to each of X, Y
        mean_x = np.mean(X, 0)
        mean_y = np.mean(Y, 0)
        cov_x = np.cov(X.T)
        [Dx, Vx] = np.linalg.eig(cov_x + 1e-3 * np.eye(d))
        Dx = np.real(Dx)
        Vx = np.real(Vx)
        # a hack in case the data are high-dimensional and the covariance matrix
        # is low rank.
        Dx[Dx <= 0] = 1e-3

        # shrink the covariance so that the drawn samples will not be so
        # far away from the data
        eig_pow = 0.9  # 1.0 = not shrink
        reduced_cov_x = Vx.dot(np.diag(Dx ** eig_pow)).dot(Vx.T) + 1e-3 * np.eye(d)
        cov_y = np.cov(Y.T)
        [Dy, Vy] = np.linalg.eig(cov_y + 1e-3 * np.eye(d))
        Vy = np.real(Vy)
        Dy = np.real(Dy)
        Dy[Dy <= 0] = 1e-3
        reduced_cov_y = Vy.dot(np.diag(Dy ** eig_pow)).dot(Vy.T) + 1e-3 * np.eye(d)
        # integer division
        Jx = int(n_test_locs // 2)
        Jy = n_test_locs - Jx

        assert Jx + Jy == n_test_locs, "total test locations is not n_test_locs"
        Tx = np.random.multivariate_normal(mean_x, reduced_cov_x, Jx)
        Ty = np.random.multivariate_normal(mean_y, reduced_cov_y, Jy)
        T0 = np.vstack((Tx, Ty))
    return T0

