import proj_split_mpi4py_sync_lasso as ps_mpi
import numpy as np


iter = 1000
n = 1000
d = 1000
A = np.random.normal(0,1,[n,d])
b = np.random.normal(0,1,n)
lam = 1e0
rho = 1e0
gamma = 1e6
Delta = 1e0

ps_mpi.ps_mpi_sync_lasso(iter,A,b,lam,rho,gamma,Delta)
