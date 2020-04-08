# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:39:09 2020

@author: pjohn
"""

import proj_split_mpi4py_sync_lasso_v1 as ps_mpi
import ps_mpi_fixed_slices as psm_fixed
import numpy as np
from mpi4py import MPI


Comm = MPI.COMM_WORLD

A = np.load('A.npy')
b = np.load('b.npy')

lam = 3e0
nslices = 10
print('lam '+str(lam))
doPlots = False      
runCVX = False 
Verbose = True
skipProjectStep = False               


#[_,s,_] = np.linalg.svd(A)
#print('largest singular value squared = '+str(s[0]**2))

iter = 1000

rho = 1e0
gamma = 1e5
adapt_gamma = False 
print('adapt gamma '+str(adapt_gamma))

Delta = 1e0

psample = 10
pid = Comm.Get_rank()



[opt_ps,z_ps] = psm_fixed.ps_mpi_sync_lasso_fixed(iter,A,b,lam,rho,gamma,Delta, 
                                                  adapt_gamma, doPlots,psample,
                                                  Comm,nslices,Verbose,skipProjectStep)





if (pid == 0) & runCVX:
    [opt_cvx,x_cvx] = ps_mpi.runCVX(A,b,lam)            
    x_cvx = np.squeeze(np.array(x_cvx))
    print("opt_ps - opt_cvx = "+str(opt_ps - opt_cvx))
    print("norm(z_ps - x_cvx)/norm(z_ps) = "+str(np.linalg.norm(z_ps - x_cvx)/np.linalg.norm(z_ps)))        
    
    
    
    
    
