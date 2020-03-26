import proj_split_mpi4py_sync_lasso as ps_mpi
import numpy as np

A = np.load('A.npy')
b = np.load('b.npy')

lam = 12.0
print('lam '+str(lam))
doPlots = True

#factor = 50
#alpha = factor*(1.0/60)
alpha = 1e0
lam = lam*alpha**2
A = alpha*A
b = alpha*b

#[_,s,_] = np.linalg.svd(A)
#print('largest singular value squared = '+str(s[0]**2))

iter = 4000

rho = 1e0
#gamma =20000.0
gamma = 30.0
adapt_gamma = "NewBalanced"
print('adapt gamma '+str(adapt_gamma))
#gamma = factor**4
#print("gamma = "+str(gamma))
Delta = 1e0
psample = -1



ps_mpi.ps_mpi_sync_lasso(iter,A,b,lam,rho,gamma,Delta,adapt_gamma, doPlots,psample)
