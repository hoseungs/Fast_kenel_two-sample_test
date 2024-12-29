library(mvtnorm)
source('func.R')


############################################################ d = 100

#################################### null

band = matrix(0, 100, 9)
for (i in 1:100) {
  X = rmvnorm(100, mean = rep(0, 100), sigma = diag(100)) 
  Y = rmvnorm(100, mean = rep(0, 100), sigma = diag(100)) 
  sigma = med_sigma(X,Y)
  band[i,][5] = sigma
  band[i,][4] = sigma - 2
  band[i,][3] = sigma - 4
  band[i,][2] = sigma - 6
  band[i,][1] = sigma - 8
  if (band[i,][1] <= 0) {
    band[i,][1]  = 0.01
  }
  band[i,][6] = sigma + 2
  band[i,][7] = sigma + 4
  band[i,][8] = sigma + 6
  band[i,][9] = sigma + 8
}

bandwidth = colMeans(band)

res100_null = rep(0, 9)
for (i in 1:100) {
  X = rmvnorm(10000, mean = rep(0, 100), sigma = diag(100)) 
  Y = rmvnorm(10000, mean = rep(0, 100), sigma = diag(100)) 
  for (j in 1:9) {
    a = fkernel(X, Y, bandwidth[j])
    if (a <= 0.05) {
      res100_null[j] = res100_null[j] + 1
    }
  }
}


#################################### mean

band = matrix(0, 100, 9)
for (i in 1:100) {
  X = rmvnorm(100, mean = rep(0, 100), sigma = diag(100)) 
  Y = rmvnorm(100, mean = rep(0.06, 100), sigma = diag(100))
  sigma = med_sigma(X,Y)
  band[i,][5] = sigma
  band[i,][4] = sigma - 2
  band[i,][3] = sigma - 4
  band[i,][2] = sigma - 6
  band[i,][1] = sigma - 8
  if (band[i,][1] <= 0) {
    band[i,][1]  = 0.01
  }
  band[i,][6] = sigma + 2
  band[i,][7] = sigma + 4
  band[i,][8] = sigma + 6
  band[i,][9] = sigma + 8
}

bandwidth = colMeans(band)

res100_mean = rep(0, 9)
for (i in 1:100) {
  X = rmvnorm(10000, mean = rep(0, 100), sigma = diag(100)) 
  Y = rmvnorm(10000, mean = rep(0.06, 100), sigma = diag(100))
  for (j in 1:9) {
    a = fkernel(X, Y, bandwidth[j])
    if (a <= 0.05) {
      res100_mean[j] = res100_mean[j] + 1
    }
  }
}


#################################### var

band = matrix(0, 100, 9)
for (i in 1:100) {
  X = rmvnorm(100, mean = rep(0, 100), sigma = diag(100)) 
  Y = rmvnorm(100, mean = rep(0, 100), sigma = 1.01*diag(100))
  sigma = med_sigma(X,Y)
  band[i,][5] = sigma
  band[i,][4] = sigma - 2
  band[i,][3] = sigma - 4
  band[i,][2] = sigma - 6
  band[i,][1] = sigma - 8
  if (band[i,][1] <= 0) {
    band[i,][1]  = 0.01
  }
  band[i,][6] = sigma + 2
  band[i,][7] = sigma + 4
  band[i,][8] = sigma + 6
  band[i,][9] = sigma + 8
}

bandwidth = colMeans(band)

res100_var = rep(0, 9)
for (i in 1:100) {
  X = rmvnorm(10000, mean = rep(0, 100), sigma = diag(100)) 
  Y = rmvnorm(10000, mean = rep(0, 100), sigma = 1.01*diag(100))
  for (j in 1:9) {
    a = fkernel(X, Y, bandwidth[j])
    if (a <= 0.05) {
      res100_var[j] = res100_var[j] + 1
    }
  }
}


############################################################ d = 200

#################################### null

band = matrix(0, 100, 9)
for (i in 1:100) {
  X = rmvnorm(100, mean = rep(0, 200), sigma = diag(200)) 
  Y = rmvnorm(100, mean = rep(0, 200), sigma = diag(200)) 
  sigma = med_sigma(X,Y)
  band[i,][5] = sigma
  band[i,][4] = sigma - 2
  band[i,][3] = sigma - 4
  band[i,][2] = sigma - 6
  band[i,][1] = sigma - 8
  if (band[i,][1] <= 0) {
    band[i,][1]  = 0.01
  }
  band[i,][6] = sigma + 2
  band[i,][7] = sigma + 4
  band[i,][8] = sigma + 6
  band[i,][9] = sigma + 8
}

bandwidth = colMeans(band)

res200_null = rep(0, 9)
for (i in 1:100) {
  X = rmvnorm(10000, mean = rep(0, 200), sigma = diag(200)) 
  Y = rmvnorm(10000, mean = rep(0, 200), sigma = diag(200)) 
  for (j in 1:9) {
    a = fkernel(X, Y, bandwidth[j])
    if (a <= 0.05) {
      res200_null[j] = res200_null[j] + 1
    }
  }
}


#################################### mean

band = matrix(0, 100, 9)
for (i in 1:100) {
  X = rmvnorm(100, mean = rep(0, 200), sigma = diag(200)) 
  Y = rmvnorm(100, mean = rep(0.025, 200), sigma = diag(200))
  sigma = med_sigma(X,Y)
  band[i,][5] = sigma
  band[i,][4] = sigma - 2
  band[i,][3] = sigma - 4
  band[i,][2] = sigma - 6
  band[i,][1] = sigma - 8
  if (band[i,][1] <= 0) {
    band[i,][1]  = 0.01
  }
  band[i,][6] = sigma + 2
  band[i,][7] = sigma + 4
  band[i,][8] = sigma + 6
  band[i,][9] = sigma + 8
}

bandwidth = colMeans(band)

res200_mean = rep(0, 9)
for (i in 1:100) {
  X = rmvnorm(10000, mean = rep(0, 200), sigma = diag(200)) 
  Y = rmvnorm(10000, mean = rep(0.025, 200), sigma = diag(200))
  for (j in 1:9) {
    a = fkernel(X, Y, bandwidth[j])
    if (a <= 0.05) {
      res200_mean[j] = res200_mean[j] + 1
    }
  }
}

#################################### var

band = matrix(0, 100, 9)
for (i in 1:100) {
  X = rmvnorm(100, mean = rep(0, 200), sigma = diag(200)) 
  Y = rmvnorm(100, mean = rep(0, 200), sigma = 1.005*diag(200))
  sigma = med_sigma(X,Y)
  band[i,][5] = sigma
  band[i,][4] = sigma - 2
  band[i,][3] = sigma - 4
  band[i,][2] = sigma - 6
  band[i,][1] = sigma - 8
  if (band[i,][1] <= 0) {
    band[i,][1]  = 0.01
  }
  band[i,][6] = sigma + 2
  band[i,][7] = sigma + 4
  band[i,][8] = sigma + 6
  band[i,][9] = sigma + 8
}

bandwidth = colMeans(band)

res200_var = rep(0, 9)
for (i in 1:100) {
  X = rmvnorm(10000, mean = rep(0, 200), sigma = diag(200)) 
  Y = rmvnorm(10000, mean = rep(0, 200), sigma = 1.005*diag(200)) 
  for (j in 1:9) {
    a = fkernel(X, Y, bandwidth[j])
    if (a <= 0.05) {
      res200_var[j] = res200_var[j] + 1
    }
  }
}


############################################################


res = list()
res$res100_null = res100_null
res$res100_mean = res100_mean
res$res100_var = res100_var
res$res200_null = res200_null
res$res200_mean = res200_mean
res$res200_var = res200_var

print(res)
