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
