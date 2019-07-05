generate_data_p0_q0 <- function(n,p,q,p0,q0){
  library(LaplacesDemon)
  library(rTensor)
  library(MASS)
  # A matrix
  x = matrix(rnorm(p*q),p,q)
  try = qr(x)
  try$rank
  A = as.matrix(qr.Q(try)[,1:p0])
  # B matrix
  x = matrix(rnorm(p*q),p,q)
  try_B = qr(x)
  try_B$rank
  B = as.matrix(qr.Q(try_B)[,1:q0])
  # beta
  beta = runif(p*q,min = -5, max=5)
  # cov of x
  U = B%*%t(B)
  V = A%*%t(A)
  var_x = kronecker(U,V)
  # mean matrix
  M = matrix(0,p*q) 
  # predictor
  data_x = matrix(0,n,p*q)
  prob = rep(0,n)
  for (i in 1:n){
    vec_x = mvrnorm(n =1, M, var_x)
    data_x[i,] = vec_x
    prob[i] = exp(sum(beta*vec_x))/(1+exp(sum(beta*vec_x)))
  }
  data_y = ifelse(prob>=0.5, 1,0)
  output = cbind(data_y,data_x)
  return(output)
}

generate_data_p_q <- function(n,p,q,n0,new_value){
  # n0 is the idex where we set the eigenvalues = 0
  # A matrix
  x = matrix(rnorm(p*q),p,q)
  try = qr(x)
  try$rank
  A = as.matrix(qr.Q(try))
  # B matrix
  x = matrix(rnorm(p*q),p,q)
  try_B = qr(x)
  try_B$rank
  B = as.matrix(qr.Q(try_B))
  # beta
  beta = runif(p*q)
  # mean matrix
  M = matrix(0,p*q) 
  # cov of x
  U =B%*%t(B)
  V =A%*%t(A)
  var_x = kronecker(U,V) # pq * pq
  # svd for var_x
  var_x_decompose = svd(var_x)
  # change the remaining eigenvalues into 0
  eigenvalue = var_x_decompose$d
  eigenvalue[n0:(p*q)] = new_value
  var_x_new = var_x_decompose$u%*%diag(eigenvalue)%*%t(var_x_decompose$v)
  # predictor
  data_x = matrix(0,n,p*q)
  prob = rep(0,n)
  for (i in 1:n){
    vec_x = mvrnorm(n =1, M, var_x_new)
    #print(vec_x[1])
    prob[i] = exp(sum(beta*vec_x))/(1+exp(sum(beta*vec_x)))
    data_x[i,] = vec_x
  }
  data_y = ifelse(prob>=0.5, 1,0)
  output = cbind(data_y,data_x)
  return(output)
}

generate_data_no_structure <- function(n,p,q){
  # beta
  beta = runif(p*q)
  var_x = diag(1,p*q)
  # mean matrix
  M = matrix(0,p*q) 
  # predictor
  data_x = matrix(0,n,p*q)
  prob = rep(0,n)
  for (i in 1:n){
    vec_x = mvrnorm(n =1, M, var_x)
    data_x[i,] = vec_x
    prob[i] = exp(sum(beta*vec_x))/(1+exp(sum(beta*vec_x)))
  }
  data_y = ifelse(prob>=0.5, 1,0)
  output = cbind(data_y,data_x)
  return(output)
}


generate_data_no_structure_cor <- function(n,p,q){
  library(Matrix)
  # beta
  beta = runif(p*q)
  var_x_no_sym = matrix(runif(p*q*p*q,0.3,0.6),p*q,p*q)
  diag(var_x_no_sym) = 1
  var_x = as.matrix(forceSymmetric(var_x_no_sym))
  var_x = as.positive.definite(var_x)
  # mean matrix
  M = matrix(0,p*q) 
  # predictor
  data_x = matrix(0,n,p*q)
  prob = rep(0,n)
  for (i in 1:n){
    vec_x = mvrnorm(n =1, M, var_x)
    data_x[i,] = vec_x
    prob[i] = exp(sum(beta*vec_x))/(1+exp(sum(beta*vec_x)))
  }
  data_y = ifelse(prob>=0.5, 1,0)
  output = cbind(data_y,data_x)
  return(output)
}
