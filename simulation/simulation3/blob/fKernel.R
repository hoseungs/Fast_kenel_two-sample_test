# main function: return p-value
fkernel = function(X, Y) {
  m = nrow(X)
  n = nrow(Y)
  
  b = floor(sqrt(mean(c(m,n))))
  
  B1 = floor(m/b)
  B2 = floor(n/b)
  
  rx = m - B1*b
  ry = n - B2*b
  
  if (rx > 0) {
    B1i = c(rep(B1, b-rx), rep(B1+1, rx))
  } else {
    B1i = rep(B1, b)
  }
  if (ry > 0) {
    B2i = c(rep(B2, b-ry), rep(B2+1, ry))
  } else {
    B2i = rep(B2, b)
  }
  
  Bi = B1i + B2i
  
  p1i = B1i*(B1i-1)/Bi/(Bi-1)
  p2i = p1i*(B1i-2)/(Bi-2)
  p3i = p2i*(B1i-3)/(Bi-3)
  
  q1i = B2i*(B2i-1)/Bi/(Bi-1)
  q2i = q1i*(B2i-2)/(Bi-2)
  q3i = q2i*(B2i-3)/(Bi-3)
  
  Zw = Zd = vector('list', 9)
  u1 = v1 = u2 = v2 = 0
  for (j in 1:b) {
    u1 = v1 + 1
    v1 = u1 + B1i[j] - 1
    u2 = v2 + 1
    v2 = u2 + B2i[j] - 1
    Xi = X[u1:v1,]
    Yi = Y[u2:v2,]
    
    sigma = med_sigma(Xi,Yi)
    
    temp = getkernel(Xi, Yi, sigma, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    
    temp1 = getkernel(Xi, Yi, sigma-2, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    temp2 = getkernel(Xi, Yi, sigma-4, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    temp3 = getkernel(Xi, Yi, sigma-6, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    if ((sigma-8)<=0) {
        temp4 = getkernel(Xi, Yi, 0.01, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    } else {
        temp4 = getkernel(Xi, Yi, sigma-8, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    }
    
    temp5 = getkernel(Xi, Yi, sigma+2, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    temp6 = getkernel(Xi, Yi, sigma+4, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    temp7 = getkernel(Xi, Yi, sigma+6, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    temp8 = getkernel(Xi, Yi, sigma+8, B1i[j], B2i[j], Bi[j], p1i[j], p2i[j], p3i[j], q1i[j], q2i[j], q3i[j])
    
    Zw[[1]] = c(Zw[[1]], temp4$Z.w)
    Zd[[1]] = c(Zd[[1]], temp4$Z.d)
    Zw[[2]] = c(Zw[[2]], temp3$Z.w)
    Zd[[2]] = c(Zd[[2]], temp3$Z.d)
    Zw[[3]] = c(Zw[[3]], temp2$Z.w)
    Zd[[3]] = c(Zd[[3]], temp2$Z.d)
    Zw[[4]] = c(Zw[[4]], temp1$Z.w)
    Zd[[4]] = c(Zd[[4]], temp1$Z.d)
    Zw[[5]] = c(Zw[[5]], temp$Z.w)
    Zd[[5]] = c(Zd[[5]], temp$Z.d)
    Zw[[6]] = c(Zw[[6]], temp5$Z.w)
    Zd[[6]] = c(Zd[[6]], temp5$Z.d)
    Zw[[7]] = c(Zw[[7]], temp6$Z.w)
    Zd[[7]] = c(Zd[[7]], temp6$Z.d)
    Zw[[8]] = c(Zw[[8]], temp7$Z.w)
    Zd[[8]] = c(Zd[[8]], temp7$Z.d)
    Zw[[9]] = c(Zw[[9]], temp8$Z.w)
    Zd[[9]] = c(Zd[[9]], temp8$Z.d)
  }
  
  p = rep(0, 9)
  for (i in 1:9) {
    Zwbar = mean(Zw[[i]])
    Zdbar = mean(Zd[[i]])
      
    pw = pnorm(-sqrt(b)*Zwbar)
    pd = 2*pnorm(-sqrt(b)*abs(Zdbar)) # two-sided
      
    temp = sort(c(pw, pd))
    p[i] = 2*min(temp[1], temp[2])
  }
  
  return(p)
}


# supporting function
getkernel = function(X, Y, sigma, B1, B2, B, p1, p2, p3, q1, q2, q3) {
  Z = rbind(X,Y)
  
  K = exp(-as.matrix(dist(Z)^2)/sigma^2/2) - diag(B)
  
  R_temp = rowSums(K)
  
  R0 = sum(K)
  R1 = sum(K^2)
  R2 = sum(R_temp^2) - R1
  R3 = R0^2 - 2*R1 - 4*R2
  
  Kx = sum(K[1:B1,1:B1])/B1/(B1-1)
  Ky = sum(K[(B1+1):B,(B1+1):B])/B2/(B2-1)
  
  mu = R0/B/(B-1)
  var_x = (2*R1*p1 + 4*R2*p2 + R3*p3)/B1/B1/(B1-1)/(B1-1) - mu^2
  var_y = (2*R1*q1 + 4*R2*q2 + R3*q3)/B2/B2/(B2-1)/(B2-1) - mu^2
  cov_x_y = R3/B/(B-1)/(B-2)/(B-3) - mu^2
  
  # test statistic Z_d
  u.d = B1*(B1-1)
  v.d = -B2*(B2-1)
  mean.d = mu*u.d + mu*v.d
  var.d = (u.d^2)*var_x + (v.d^2)*var_y + 2*u.d*v.d*cov_x_y
  Z.d = (Kx*u.d + Ky*v.d - mean.d)/sqrt(var.d)
  
  # test statistic Z_w
  u.w = B1/B
  v.w = B2/B
  mean.w = mu*u.w + mu*v.w
  var.w = var_x*u.w^2 + var_y*v.w^2 + 2*u.w*v.w*cov_x_y
  Z.w = (Kx*u.w + Ky*v.w - mean.w)/sqrt(var.w)
  
  res = list()
  res$Z.w = Z.w
  res$Z.d = Z.d
  
  return(res)
}

# supporting function
med_sigma = function(X, Y) {
  aggre = rbind(X,Y)
  med = median(dist(aggre)^2)
  return(sqrt(med))
}
