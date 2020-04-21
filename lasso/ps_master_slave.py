# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:39:12 2020

@author: pjohn

Implementation of projective splitting using a naive master-slave 
topology, just to test things out. The master will be in charge of performing 
the projections, while the slaves will work on the block updates using 
forward steps. This is applied to the lasso problem. 
there will be a synchronous and an asynchronous version of the master. The slaves
are inherrently synchronous: they just wait for new data from the master...
"""

import numpy as np
import proj_split_mpi4py_sync_lasso as psmpi
import ps_lasso as ls
from mpi4py import MPI
import sys 
from matplotlib import pyplot as plt


def ps_sync_masterslave(Comm,A,b,nslices,Verbose,rho0,Delta,lam,doPlots,gamma,iterations):
    
    
    pid = Comm.Get_rank() # processor ID
    nprocs  = Comm.Get_size() # number of processors 
    (n,d) = A.shape
    nslaves = nprocs - 1
    if(nslaves == 0):
        print("Error: Need more than 2 nodes for master-slave, exiting")
        sys.exit()
    
    if (pid == 0) & Verbose:
        print("Number of processors = "+str(nprocs))
        print("Number of slices = "    +str(nslices))
        print("Number of total measurements = "+str(n))
        print("Number of features/elements = "+str(d))

    part_ind = ls.create_simple_partition(n,nslices) # partitition the indices 0...n-1 into nslices groups
    part_proc = ls.create_simple_partition(nslices,nslaves) # partitition the slices into nslaves groups
    
    # count and displacements are used in the scatter/gather communications
    count = [0]
    displacements = [0]
    for i in range(1,len(part_proc)+1):
        count.append((part_proc[i-1][1]-part_proc[i-1][0])*d)
        displacements.append(sum(count[0:i]))
    
    # every processor needs z 
    z = np.zeros(d)
    
    if pid>0:
        pid_num_slices = part_proc[pid-1][1]-part_proc[pid-1][0] # note pid-1 not pid 
        wLoc = np.zeros((pid_num_slices,d))   
        xLoc = np.zeros((pid_num_slices,d))           
        x = np.zeros([])
        y = np.zeros((pid_num_slices,d))
        w = np.zeros([])
        rho = np.ones(pid_num_slices)*rho0 
        local_data = np.zeros(2*d+2) # will be the local sum of the local variables
        global_data = np.zeros(2*d+2)
    else:
        w = np.zeros((nslices,d))
        x = np.zeros((nslices,d))
        xLoc = np.zeros([])
        wLoc = np.zeros([])
        global_data = np.zeros(2*d+2) # the root node stores (sumy,sumx,phi,sum_norm_xi_sq) 
        local_data = np.zeros(2*d+2)
        

    # local and global data are 2d+2 length buffers storing y,x,phi, and norm xi sq   
    # below are the indices for each data segment
    ind_y = (0,d)
    ind_x = (d,2*d)
    ind_phi = 2*d
    ind_norm_xi_sq = 2*d+1
    
        
    
    # processor 0 will plot results
    if (pid == 0) & Verbose:
        normGrads = []
        phis = []
        func_vals = []
        print_freq = int(iterations/10)
        if(print_freq==0):
            print_freq+=1
        
        
    
    for k in range(iterations):
        if (pid == 0) & Verbose:
            # Processor 0 will print the iteration number
            if k%(print_freq) == 0:
                print("iter "+str(k))
                
                
        # root node sends z to all the slaves 
        Comm.Bcast(z,root=0)
        
        # root node sends the appropriate w to each slave. Scatterv because
        # variable amounts. Each slave might have a different number of w. 
        
        Comm.Scatterv([w,count,displacements,MPI.DOUBLE],wLoc)
                    
        # slaves compute block updates
        if pid != 0:
            
            [xLoc,rho,local_data,y] = multiBlockUpdate(part_proc[pid-1][0],pid_num_slices,
                                                     xLoc,rho,z, wLoc,A,part_ind,b,lam,
                                                     nslices,Delta,local_data,ind_y,
                                                     ind_x,ind_phi,ind_norm_xi_sq,y)
        
        # Perform a global reduce to get data to the root node. 
        Comm.Reduce(local_data,global_data,root = 0)  
             
        # Perform a Gatherv to get x at the root node. 
        # Gatherv because there are variable amounts of data at each slave. 
        Comm.Gatherv(xLoc,[x,count,displacements,MPI.DOUBLE])
      
                
        # Perform hplane projection
        if pid == 0:                          
            [z,w,normGrad,phi] = hplaneProject(global_data,ind_x,nslices,ind_y,
                                               ind_norm_xi_sq,gamma,ind_phi,z,w,x)
            
            if doPlots:
                normGrads.append(normGrad)
                phis.append(phi)                
                func_vals.append(psmpi.lasso_val(z,A,b,lam))
        
    if pid == 0:
        print("-------------------------------------------")
        nnz = sum(abs(z)>1e-10)
        print("z nnz = "+str(nnz))    
        if doPlots:
            print("final function value "+str(func_vals[-1]))                      
            plt_iter = iterations
            fig,ax = plt.subplots(2,2)
            ax[0,0].plot(func_vals[0:plt_iter])
            ax[0,0].set_title('function values')
            ax[0,1].semilogy(range(plt_iter) , np.array(func_vals[0:plt_iter]) - func_vals[-1])
            ax[0,1].set_title('function vals')
            ax[1,0].semilogy(phis[0:plt_iter])
            ax[1,0].set_title("phis (should be positive)")            
            ax[1,1].semilogy(normGrads)
            ax[1,1].set_title("norm of hplane grads")                        
            plt.show()
    
            return func_vals[-1]
        
            

def multiBlockUpdate(start_slice,pid_num_slices,x,rho,z,w,A,part_ind,b,lam,
                     nslices,Delta,local_data,ind_y,ind_x,ind_phi,ind_norm_xi_sq,y):
    
    for i in range(pid_num_slices):
            
        # block update using each processor's slice of the data.
        
       
        this_slice = start_slice + i
        
        [x[i],y[i],rho[i]] = \
            psmpi.update_block(z,w[i],rho[i],A[part_ind[this_slice][0]:part_ind[this_slice][1]],\
                         b[part_ind[this_slice][0]:part_ind[this_slice][1]],lam/nslices,Delta)
                                           
        if i == 0:
            # first slice, rewrite local data 
            local_data[ind_y[0]:ind_y[1]] = y[i]
            local_data[ind_x[0]:ind_x[1]] = x[i]                
            local_data[ind_phi] = (z - x[i]).dot(y[i] - w[i])
            local_data[ind_norm_xi_sq] = np.linalg.norm(x[i],2)**2    
        else:
            # keep track of sum 
            local_data[ind_x[0]:ind_x[1]] += x[i]
            local_data[ind_y[0]:ind_y[1]] += y[i]
            local_data[ind_phi] += (z - x[i]).dot(y[i] - w[i])
            local_data[ind_norm_xi_sq] += np.linalg.norm(x[i],2)**2 
            
    return [x,rho,local_data,y]
    
def hplaneProject(global_data,ind_x,nslices,ind_y,ind_norm_xi_sq,gamma,ind_phi,z,w,x):
    global_data[ind_x[0]:ind_x[1]]  = (1.0/nslices)*global_data[ind_x[0]:ind_x[1]] # this is xbar, the average over xi for all slices  

    norm_sumy_sq = np.linalg.norm(global_data[ind_y[0]:ind_y[1]],2)**2
    sum_norm_ui_sq = global_data[ind_norm_xi_sq] - nslices*np.linalg.norm(global_data[ind_x[0]:ind_x[1]],2)**2    
        
    normGrad = gamma**(-1)*norm_sumy_sq + sum_norm_ui_sq
    phi = global_data[ind_phi]

    
    if abs(normGrad)>1e-20:
        z = z - (gamma**(-1)*phi/normGrad)*global_data[ind_y[0]:ind_y[1]]
        w = w - (phi/normGrad)*(x - global_data[ind_x[0]:ind_x[1]])   

             
    
    return [z,w,normGrad,phi]
        

