library(reticulate)
source('fKernel.R')

# H1

x = vector('list', 500)
y = vector('list', 500)

for (i in 1:500) {
  x[[i]] = py$Blobs(as.integer(10000))[[1]]
  y[[i]] = py$Blobs(as.integer(10000))[[2]]
}

res = matrix(0, 500, 9)

for (i in 1:500) {
    X = x[[i]]
    Y = y[[i]]
    
    a = fkernel(X, Y)
    
    for (j in 1:9) {
        if (a[j] <= 0.05) {
            res[i,j] = res[i,j] + 1
        }
    }
}

res = apply(res,2,sum)


# Null

x = vector('list', 500)
y = vector('list', 500)

for (i in 1:500) {
    x[[i]] = py$Blobsnull(as.integer(10000))[[1]]
    y[[i]] = py$Blobsnull(as.integer(10000))[[2]]
}

res = matrix(0, 500, 9)

for (i in 1:500) {
    X = x[[i]]
    Y = y[[i]]
    
    a = fkernel(X, Y)
    
    for (j in 1:9) {
        if (a[j] <= 0.05) {
            res[i,j] = res[i,j] + 1
        }
    }
}

res = apply(res,2,sum)
