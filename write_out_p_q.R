rm(list=ls())
source("generate_data.R")
# generate data
true_p = 20
true_q = 20
new_eigen = 0.01
start_point = 200
x = generate_data_p_q(525,true_p,true_q,start_point,new_eigen)
write.csv(x,"data_525_p=20_q=20_eigen_0.01.csv")
# turn into tensor data
predictors = x[1:525,-1] # 500*400
predictors_tensor = array(0, dim = c(true_p,true_q,525))
for (obs in 1:525){
  for (q in 1:true_p){
    for (p in 1:true_q){
      predictors_tensor[p,q,obs] = predictors[obs,p+20*(q-1)]
    }
  }
}

train_x_tensor = predictors_tensor[,,1:500]
test_x_tensor = predictors_tensor[,,501:525]

mpca_x=mpca(as.tensor(train_x_tensor),ranks=c(p,q),max_iter = 1000, tol=1e-1)
mpca_x$conv
temp_A=mpca_x$U[[1]]
temp_B=mpca_x$U[[2]]
# write out the estimated U for training data
U_p_q_train = array(0,dim = c(p,q,500))
vec_U_train = matrix(0,500,p*q)
for (i in 1:500){
  U_p_q_train[,,i] = t(temp_A)%*%attributes(mpca_x$est)$data[,,i]%*%temp_B
  vec_U_train[i,] = as.vector(U_p_q_train[,,i])
}
# reduce test data
U_p_q_test = array(0,dim = c(p,q,25))
vec_U_test = matrix(0,25,p*q)
for (i in 1:25){
  U_p_q_test[,,i] = t(temp_A)%*%test_x_tensor[,,i]%*%temp_B
  vec_U_test[i,] = as.vector(U_p_q_test[,,i])
}
output_x = rbind(vec_U_train,vec_U_test)
output_reduced = cbind(x[,1],output_x)
write.csv(output_reduced,paste("p0q020_data_525_p=",p,"_q=",q,"_new_eigen=",new_eigen,".csv",sep = ""))

# reduce
p0 = 4
q0 = 4
mpca_x=mpca(as.tensor(train_x_tensor),ranks=c(p0,q0),max_iter = 1000, tol=1e-1)
mpca_x$conv
temp_A=mpca_x$U[[1]]
temp_B=mpca_x$U[[2]]
# write out the estimated U for training data
U_p0_q0_train = array(0,dim = c(p0,q0,500))
vec_U_train = matrix(0,500,p0*q0)
for (i in 1:500){
  U_p0_q0_train[,,i] = t(temp_A)%*%attributes(mpca_x$est)$data[,,i]%*%temp_B
  vec_U_train[i,] = as.vector(U_p0_q0_train[,,i])
}
# reduce test data
U_p0_q0_test = array(0,dim = c(p0,q0,25))
vec_U_test = matrix(0,25,p0*q0)
for (i in 1:25){
  U_p0_q0_test[,,i] = t(temp_A)%*%test_x_tensor[,,i]%*%temp_B
  vec_U_test[i,] = as.vector(U_p0_q0_test[,,i])
}
output_x = rbind(vec_U_train,vec_U_test)
output_reduced = cbind(x[,1],output_x)
write.csv(output_reduced,paste("p0q04_data_525_p=",p,"_q=",q,"_new_eigen=",new_eigen,".csv",sep = ""))
