import proj_split_mpi4py_sync_lasso as ps_mpi
import numpy as np

A = np.load('A.npy')
b = np.load('b.npy')
print(A.shape)
print(b.shape)
iter = 1000
lam = 1e0
rho = 1e0
gamma = 1e6
Delta = 1e0

ps_mpi.ps_mpi_sync_lasso(iter,A,b,lam,rho,gamma,Delta)
