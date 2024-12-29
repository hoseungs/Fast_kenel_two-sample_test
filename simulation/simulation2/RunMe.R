source('fKernel.R')

library(mvtnorm)
library(ade4)
library(gTests)
library(Ecume)
library(Ball)

Sig = function(val, d) {
  cov = matrix(0, d, d)
  for (i in 1:d) {
    for (j in 1:d) {
      cov[i,j] = val^(abs(i-j))
    }
  }
  return(cov)
}

dd = 100 # dimension
m = 5000 # m=n

A = Sig(0.4,dd) # covariance structure
B = Sig(0.4,dd)

# location alternatives
X = exp(rmvnorm(m, mean = rep(0, dd), sigma = A))
Y = exp(rmvnorm(m, mean = rep(0.05, dd), sigma = B))

# New
a = fkernel(X, Y)

# GT
E = mstree(dist(rbind(X,Y)))
a = g.tests(E, 1:m, (m+1):(2*m), test.type="g")

# BT
a = bd.test(X, Y, method = "limit")

# CT
X = exp(rmvnorm(2*m, mean = rep(0, dd), sigma = A))
Y = exp(rmvnorm(2*m, mean = rep(0.05, dd), sigma = B))
a = classifier_test(X, Y, split = 0.5)
