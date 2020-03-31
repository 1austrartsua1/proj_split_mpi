# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:28:33 2020

@author: pjohn
"""

import numpy as np
import ps_lasso as ls
from matplotlib import pyplot as plt
import proj_split_mpi4py_sync_lasso_v1 as psmpi



def ps_mpi_sync_lasso_fixed(iter,A,b,lam,rho,gamma,Delta,adapt_gamma, doPlots,p,Comm,nslices,Verbose):
    '''
    projective splitting applied to the lasso problem. This is a synchronous parallel
    version using MPI.
    This version takes in a number of sliced as an input parameter and then distributes 
    the work to the processors. 
    '''
    
    pid = Comm.Get_rank() # processor ID
    nprocs  = Comm.Get_size() # number of processors 
    (n,d) = A.shape
    
    if (pid == 0) & Verbose:
        print("Number of processors = "+str(nprocs))
        print("Number of slices = "    +str(nslices))
        print("Number of total measurements = "+str(n))
        print("Number of features/elements = "+str(d))
    
    
    
    # The number of slices is the input parameter nslices.
    # We can subdivide the data matrix up into nslices.
    
    part_ind = ls.create_simple_partition(n,nslices)
    # partition of the rows 0... n-1 into nslices slices. 
    # if nslices does not divide n, the final slice is enlargened to include the remainder
    
    # We need another partition for which slices are assigned to which processors 
    # partition of slices 0...nslices-1 into nprocs groups
    # if nprocs does not divide nslices, the last processor is tasked with a smaller number of slices 
    # and all other processors have an equal and larger number of slices
    part_proc = ls.create_simple_partition(nslices,nprocs,option="small_last")
    
    if pid == 0:
        if adapt_gamma == "Sample":
            if p != -1:
                # estimate the primal-dual scaling by sampling p points and measuring ratio
                # of primal norm to dual norm. If p equals -1, don't do this.
                i = part_proc[pid][0] # simply use the first slice assigned to the 1st processor for this estimate 
                
                av_ratio_sq = \
                    ls.estimate_pd_scale(A[part_ind[i][0]:part_ind[i][1]],b[part_ind[i][0]:part_ind[i][1]],lam/nslices,p)
                print("P"+str(i)+" the average ratio squared from "+str(p)+" points is "+str(av_ratio_sq))
                #gamma = (1.0/size)*Comm.Allreduce(av_ratio_sq)
                gamma = av_ratio_sq
                print('gamma via sample = '+str(gamma))


    if adapt_gamma == "Sample":
        Comm.Bcast(gamma,root = 0) # XX incorrect 

    pid_num_slices = part_proc[pid][1]-part_proc[pid][0]
    if(Verbose):
        print("Processor "+str(pid)+" is in charge of "+str(pid_num_slices)+" slices")

    # each processor has its own wis for all i in its slices, and a local redundant copy of z 
    # also needs to keep track of x[i] for all i in its slices
    z = np.zeros(d)
    w = np.zeros((pid_num_slices,d))   
    x = np.zeros((pid_num_slices,d))   
    
    
    # Each processor computes the projection onto the hyperplane to get the 
    # next z and w. This is slightly redundant because we do z on each processor, but saves communicating 
    # z and w_i and since the hyperplane projection computation is lightweight, 
    # it makes sense to do it on each processor and save on communication. 
    # To compute the projection, we need sum_i y_i, sum_i x_i, sum_i phi_i and sum_i norm(x_i,2)**2
    # All four of these are computed in a single Allreduce operation. 
    # local variables are sums of local x_i, y_i, phi_i, and normxi_sq. These are stored in 
    # the buffer local_data. The global variables which are computed by reduction 
    # are stored in global_data. Here is the format:  
    
    #sumy          = global_data[0:d]
    #xav           = global_data[d:2*d]
    #phi           = global_data[2*d]
    #sum_normxi_sq = global_data[2*d+1]
    
    ind_y = (0,d)
    ind_x = (d,2*d)
    ind_phi = 2*d
    ind_norm_xi_sq = 2*d+1
    
    global_data = np.empty(2*d+2) # the processor's own local copy of the globally shared data (sumy,sumx,phi,sum_norm_xi_sq)    
    local_data = np.zeros(2*d+2) # will be the local sum of the local variables 
    
    
    if adapt_gamma == "residBalanced":
        # These are needed for the first run of the resid balanced gamma procedure 
        sum_norm_ui_sq = 0.0
        norm_sumy_sq = 0.0
        
    # processor 0 will plot results
    if (pid == 0) & Verbose:
        normGrads = []
        phis = []
        func_vals = []
        print_freq = int(iter/10)
        ratio_gradz2w = []

    

    for k in range(iter):
        if (pid == 0) & Verbose:
            # Processor 0 will print the iteration number
            if k%(print_freq) == 0:
                print("iter "+str(k))
                

        if adapt_gamma == "residBalanced":
                gamma = psmpi.resid_balance(gamma,norm_sumy_sq,sum_norm_ui_sq,nslices)


        # Now loop over all slices under the domain of control of this processor 
        for i in range(pid_num_slices):
        
            # block update using each processor's slice of the data.
            this_slice = part_proc[pid][0] + i
            [x[i],newy,rho] = \
                psmpi.update_block(z,w[i],rho,A[part_ind[this_slice][0]:part_ind[this_slice][1]],\
                             b[part_ind[this_slice][0]:part_ind[this_slice][1]],lam/nslices,Delta)
            if i == 0:
                # first slice, rewrite local data 
                local_data[ind_x[0]:ind_x[1]] = x[i]
                local_data[ind_y[0]:ind_y[1]] = newy
                local_data[ind_phi] = (z - x[i]).dot(newy - w[i])
                local_data[ind_norm_xi_sq] = np.linalg.norm(x[i],2)**2    
            else:
                # keep track of sum 
                local_data[ind_x[0]:ind_x[1]] += x[i]
                local_data[ind_y[0]:ind_y[1]] += newy
                local_data[ind_phi] += (z - x[i]).dot(newy - w[i])
                local_data[ind_norm_xi_sq] += np.linalg.norm(x[i],2)**2  
            
        
        # projection updates
        # in this version, we are using the simplified paper formulation of the hyperplane                    
        # which uses the average over the xis. This is more convenient for a symmetric 
        # distributed implementation, easier to put in SPMD form. 
        # A single allreduce on the 2*d+2 NumPy buffer to get global_data from local_data 
        Comm.Allreduce(local_data,global_data)
        
        global_data[ind_x[0]:ind_x[1]]  = (1.0/nslices)*global_data[ind_x[0]:ind_x[1]] # this is xbar, the average over xi for all slices  

        norm_sumy_sq = np.linalg.norm(global_data[ind_y[0]:ind_y[1]],2)**2
        sum_norm_ui_sq = global_data[ind_norm_xi_sq] - nslices*np.linalg.norm(global_data[ind_x[0]:ind_x[1]],2)**2
        
        
        
        if adapt_gamma == "Lipschitz":
            Li = 1.0/rho # local estimate for the Lipschitz constant of this loss slice
            gammai = Li**2 # local adaptive choice for the primal-dual scaling parameter
            Comm.Allreduce(gammai,gamma) # XX incorrect to choose a gamma, take the average across all choices for each block
            gamma = gamma/nslices 
        elif adapt_gamma == "OldBalanced":
            if abs(sum_norm_ui_sq)>1e-20:
                gamma =  np.sqrt(nslices*norm_sumy_sq/sum_norm_ui_sq) # this is the original version
        elif adapt_gamma == "NewBalanced":
            if abs(sum_norm_ui_sq)>1e-20:
                gamma =  (nslices*norm_sumy_sq/sum_norm_ui_sq) # new version using eta conversion.


        normGrad = gamma**(-1)*norm_sumy_sq + sum_norm_ui_sq
        phi = global_data[ind_phi]

        if abs(normGrad)>1e-20:
            z = z - (gamma**(-1)*phi/normGrad)*global_data[ind_y[0]:ind_y[1]]
            w = w - (phi/normGrad)*(x - global_data[ind_x[0]:ind_x[1]])
        


        if (pid == 0) & Verbose:
            normGrads.append(normGrad)
            phis.append(phi)
            if doPlots:
                func_vals.append(psmpi.lasso_val(z,A,b,lam))            
                
            ratio_gradz2w.append(norm_sumy_sq/sum_norm_ui_sq)
           
    
    
    

    if (pid == 0) & Verbose:
        print("-------------------------------------------")
        nnz = sum(abs(z)>1e-10)
        print("z nnz = "+str(nnz))        
        
        if doPlots & Verbose:
            print("final function value "+str(func_vals[-1]))
            plt_iter = iter
            fig,ax = plt.subplots(2,2)
            ax[0,0].plot(func_vals[0:plt_iter])
            ax[0,0].set_title('function values')
            ax[0,1].semilogy(range(plt_iter) , np.array(func_vals[0:plt_iter]) - func_vals[-1])
            ax[0,1].set_title('function vals')
            ax[1,0].semilogy(phis[0:plt_iter])
            ax[1,0].set_title("phis (should be positive)")
            ax[1,1].plot(ratio_gradz2w)
            ax[1,1].set_title('gradz/gradw')
            #plt.semilogy(normGrads)
            #plt.title("norm of gradients of phi")
            plt.show()
    if (pid == 0) & Verbose:
        return [func_vals[-1],z]        
    else:
        return [0.0,z]