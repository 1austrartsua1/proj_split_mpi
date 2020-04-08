# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:15:05 2020

@author: pjohn
"""
doPlots = False 
Verbose = True
runCVX = False  



import proj_split_mpi4py_sync_lasso_v1 as ps_mpi
import ps_master_slave as ps
import numpy as np
from mpi4py import MPI


Comm = MPI.COMM_WORLD
pid = Comm.Get_rank()

A = np.load('A.npy')
b = np.load('b.npy')

lam = 3e1
nslices = 10
print('lam '+str(lam))



iterations = 1000
rho0 = 1e0
gamma = 1e6
Delta = 1e0

if (pid == 0):
    
    if runCVX:
        opt_ps = ps.ps_sync_masterslave(Comm,A,b,nslices,Verbose,rho0,Delta,lam,doPlots,gamma,iterations)
        [opt_cvx,x_cvx] = ps_mpi.runCVX(A,b,lam)            
        x_cvx = np.squeeze(np.array(x_cvx))
        print("opt_ps - opt_cvx = "+str(opt_ps - opt_cvx))
        print('cvx nonzeros = '+str(sum(abs(x_cvx)>1e-6)))
    else:
        ps.ps_sync_masterslave(Comm,A,b,nslices,Verbose,rho0,Delta,lam,doPlots,gamma,iterations)
else:
    ps.ps_sync_masterslave(Comm,A,b,nslices,Verbose,rho0,Delta,lam,doPlots,gamma,iterations)
    
    
    