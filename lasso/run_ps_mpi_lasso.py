import proj_split_mpi4py_sync_lasso as ps_mpi
import numpy as np

A = np.load('A.npy')
b = np.load('b.npy')

lam = 5e-2

#factor = 50
#alpha = factor*(1.0/60)
alpha = 1e0
lam = lam*alpha**2
A = alpha*A
b = alpha*b

#[_,s,_] = np.linalg.svd(A)
#print('largest singular value squared = '+str(s[0]**2))

iter = 1000

rho = 1e0
gamma = 1e0
adapt_gamma = True
#gamma = factor**4
#print("gamma = "+str(gamma))
Delta = 1e0

ps_mpi.ps_mpi_sync_lasso(iter,A,b,lam,rho,gamma,Delta,adapt_gamma)
