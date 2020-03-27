import proj_split_mpi4py_sync_lasso as ps_mpi
import numpy as np

A = np.load('A.npy')
b = np.load('b.npy')

lam = 10.0
print('lam '+str(lam))
doPlots = True


#[_,s,_] = np.linalg.svd(A)
#print('largest singular value squared = '+str(s[0]**2))

iter = 1000

rho = 1e0
gamma = 1e4
adapt_gamma = False
print('adapt gamma '+str(adapt_gamma))

Delta = 1e0
psample = -1



ps_mpi.ps_mpi_sync_lasso(iter,A,b,lam,rho,gamma,Delta,adapt_gamma, doPlots,psample)
