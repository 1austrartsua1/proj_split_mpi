import proj_split_mpi4py_sync_lasso_v1 as ps_mpi
import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

Comm = MPI.COMM_WORLD

A = np.load('A.npy')
b = np.load('b.npy')

lam = 3e0
print('lam '+str(lam))
doPlots = False 


#[_,s,_] = np.linalg.svd(A)
#print('largest singular value squared = '+str(s[0]**2))

iter = 2000

rho = 1e0
gamma = 1e4
adapt_gamma = False 
print('adapt gamma '+str(adapt_gamma))

Delta = 1e0
psample = 10



[opt_ps,z_ps] = ps_mpi.ps_mpi_sync_lasso(iter,A,b,lam,rho,gamma,Delta,adapt_gamma, doPlots,psample,Comm)

i = Comm.Get_rank()

runCVX = True  
if (i == 0) & runCVX:
    [opt_cvx,x_cvx] = ps_mpi.runCVX(A,b,lam)            
    x_cvx = np.squeeze(np.array(x_cvx))
    print("opt_ps - opt_cvx = "+str(opt_ps - opt_cvx))
    print("norm(z_ps - x_cvx)/norm(z_ps) = "+str(np.linalg.norm(z_ps - x_cvx)/np.linalg.norm(z_ps)))        
    
    
    
    
    
