
import numpy as np
import fKernel as fk
import ME_full as me
import SCF_full as scf
import HotellingT2 as hotelling
import MMD as mmd


alpha = 0.01
J = 5


dim = 100 # dimension
num_samples = 1000 # m = n

iteration = 500 # number of simulation runs


X = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 2 * num_samples)

my = 0.8 # mean change
mean_shift = np.zeros(dim)
mean_shift[0] = my
Y = np.random.multivariate_normal(mean_shift, np.eye(dim), 2 * num_samples)


Itr = np.zeros(2 * num_samples, dtype=bool)
tr_ind = np.random.choice(2 * num_samples, int(num_samples), replace=False)
Itr[tr_ind] = True
Ite = np.logical_not(Itr)
        
X_tr, Y_tr = X[Itr, :], Y[Itr, :]
X_te, Y_te = X[Ite, :], Y[Ite, :]
        
# New
test = fk.fkernel(X_te, Y_te)
    
# ME_full
test_locs, gwidth2 = me.MEopt(X_tr, Y_tr, alpha, J)
test = me.MEtest(X_te, Y_te, test_locs, gwidth2, alpha)
    
# SCF_full
test_locs, gwidth2 = scf.SCFopt(X_tr, Y_tr, alpha, J)
test = scf.SCFtest(X_te, Y_te, test_locs, gwidth2, alpha)
    
# MMD_B
gwidth = mmd.MMD_grid_search_gwidth(X_tr, Y_tr, np.sqrt(num_samples), alpha)
test = mmd.MMDtest(X_te, Y_te, gwidth, np.sqrt(num_samples), alpha)
    
# Hotelling
hotelling.Hotelling(X_te, Y_te, alpha)
