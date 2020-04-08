# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:08:21 2020

@author: pjohn
"""
# A simple synchronous parallel implementation of FISTA with backtracking.
# Basically the gradient is computed in parallel on each different processor
# and then the results are combined with one single all-reduce operation
# Function values used in backtracking are also computed locally on each processor
# and then combined with all-reduce operations. 

runCVX = False  
doPlots = True   


import numpy as np
from mpi4py import MPI
import proj_split_mpi4py_sync_lasso_v1 as psmpi
import ps_lasso as ls 
from matplotlib import pyplot as plt

Comm = MPI.COMM_WORLD
nprocs = Comm.Get_size()
pid = Comm.Get_rank()

A = np.load('A.npy')
b = np.load('b.npy')

(n,d) = A.shape 

part_ind = ls.create_simple_partition(n,nprocs)

lam = 3e0

print('lam '+str(lam))

iter = 1000

rho = 1e0

x = np.zeros(d)
xold = np.zeros(d)
y = np.zeros(d)
g = np.zeros(d)
floc = np.zeros(1)
f = np.zeros(1)
fy = np.zeros(1)
fy_loc = np.zeros(1)
t = 1.0
if pid == 0:
    Fs = []
    backsteps = []



for k in range(iter):
    if (pid == 0) & (k%100 == 0):
        print("Fista iter = "+str(k))
    
    # Compute f(y) where f is the smooth part, ls part. 
    Alocy = A[part_ind[pid][0]:part_ind[pid][1]].dot(y)
    fy_loc[0] = 0.5*np.linalg.norm(Alocy - b[part_ind[pid][0]:part_ind[pid][1]],2)**2    
    Comm.Allreduce(fy_loc,fy)
    # compute the local gradient at y 
    gloc = A[part_ind[pid][0]:part_ind[pid][1]].T.dot(Alocy- b[part_ind[pid][0]:part_ind[pid][1]])
    
    # get the global gradient via an all-reduce 
    Comm.Allreduce(gloc,g)
    
    # backtracking procedure
    finishBT = False
    if pid == 0:
        countBT = 0
    while finishBT ==  False:
        xnew = psmpi.proxL1(y - rho*g,rho*lam)
        floc[0] = 0.5*np.linalg.norm(A[part_ind[pid][0]:part_ind[pid][1]].dot(xnew)
                                                    - b[part_ind[pid][0]:part_ind[pid][1]],2)**2
        Comm.Allreduce(floc,f)
        Qy = fy[0] + g.T.dot(xnew - y) + 0.5*rho**(-1)*np.linalg.norm(xnew-y,2)**2
        if floc[0]<= Qy:
            finishBT = True
        else:
            rho = 0.5*rho 
        if pid == 0:
            countBT+=1
            
    # compute the new momentum parameter
    tnew = 0.5 + 0.5 * np.sqrt(1 + 4 * t**2)
    beta = (t - 1) / tnew
    t = tnew
    
    #update x variables 
    xold = x
    x = xnew
    # momentum update 
    y = x + beta*(x-xold)
    
    if (pid == 0) & doPlots:
        Fs.append(psmpi.lasso_val(x,A,b,lam))
        backsteps.append(countBT)
        
if pid == 0:
    if runCVX:
        [opt_cvx,x_cvx] = psmpi.runCVX(A,b,lam)            
        x_cvx = np.squeeze(np.array(x_cvx))
        print("opt_fista - opt_cvx = "+str(Fs[-1] - opt_cvx))
        print("norm(xfista - x_cvx)/norm(xfista) = "+str(np.linalg.norm(x - x_cvx)/np.linalg.norm(x)))        
    if doPlots:
        plt.semilogy(Fs - Fs[-1])
        plt.title("parallel fista func vals")
        plt.show()
        plt.plot(backsteps)
        plt.title('fista stepsizes')
        plt.show()

    
    
    
        
    
    
    
    
    
    
    
    
    
    
    