from mpi4py import MPI
import numpy as np
import ps_lasso as ls
from matplotlib import pyplot as plt


def ps_mpi_sync_lasso(iter,A,b,lam,rho,gamma,Delta,adapt_gamma, doPlots,p):
    '''
    projective splitting applied to the lasso problem. This is a synchronous parallel
    version using MPI.
    '''

    Comm = MPI.COMM_WORLD
    i = Comm.Get_rank()
    size  = Comm.Get_size()

    # now that we know the number of processors (size) and the rank,
    # we can subdivide the data matrix appropriately.
    if i == 0:
        print("Number of processors (one per slice) = "+str(size))
    (n,d) = A.shape
    partition = ls.create_simple_partition(n,size)
    # if size does not divide n, the final slice is enlargened to include the remainder
    if i == 0:
        if adapt_gamma == "Sample":
            if p != -1:
                # estimate the primal-dual scaling by sampling p points and measuring ratio
                # of primal norm to dual norm. If p equals -1, don't do this.
                av_ratio_sq = ls.estimate_pd_scale(A[partition[i]],b[partition[i]],lam/size,p)
                print("P"+str(i)+" the average ratio squared from "+str(p)+" points is "+str(av_ratio_sq))
                #gamma = (1.0/size)*Comm.Allreduce(av_ratio_sq)
                gamma = av_ratio_sq
                print('gamma via sample = '+str(gamma))


    #Comm.Bcast(gamma,root = 0)

    print("P"+str(i)+" gamma = "+str(gamma))

    # each processor has its own local redundant copy of z
    z = np.zeros(d)

    # each processor has its own local (xi,yi) and wi
    wi = np.zeros(d)
    xi = np.zeros(d)
    yi = np.zeros(d)
    
    # each processor needs empty variables to store phi, sumy
    phi = np.empty(1)
    norm_ui_sq = np.empty(1)
    phi_i = np.empty(1)
    sumy = np.empty(d)
    xav = np.empty(d)
    sum_norm_ui_sq = np.zeros(1)
    
    # processor 0 will plot results
    if i == 0:
        normGrads = []
        phis = []
        func_vals = []

        print_freq = int(iter/10)
        ratio_gradz2w = []

    norm_sumy_sq = 0.0

    for k in range(iter):
        if i == 0:
            # Processor 0 will print the iteration number
            if k%(print_freq) == 0:
                print("iter "+str(k))
                print("gamma = "+str(gamma))

        if adapt_gamma == "residBalanced":
                gamma = resid_balance(gamma,norm_sumy_sq,sum_norm_ui_sq[0],size)

        # block update using each processor's slice of the data.
        [xi,yi,rho] = update_block(z,wi,rho,A[partition[i]],b[partition[i]],lam/size,Delta)
            


        # projection updates
        phi_i[0] = (z - xi).dot(yi - wi) # local contribution to the affine function phi
        Comm.Allreduce(phi_i,phi) # scalar all reduce, summation
        Comm.Allreduce(yi,sumy)   # vector all reduce, summation
        norm_sumy_sq = np.linalg.norm(sumy,2)**2
        Comm.Allreduce(xi,xav) 
        xav = (1.0/size)*xav # vector all reduce, summation (average)
        ui = xi - xav # in this version, we are using the simplified paper formulation of the hyperplane
        norm_ui_sq[0] = np.linalg.norm(ui,2)**2
        Comm.Allreduce(norm_ui_sq,sum_norm_ui_sq) # scalar all reduce



        if adapt_gamma == "Lipschitz":
            Li = 1.0/rho # local estimate for the Lipschitz constant of this loss slice
            gammai = Li**2 # local adaptive choice for the primal-dual scaling parameter
            Comm.Allreduce(gammai,gamma) # to choose a gamma, take the average across all choices for each block
            gamma = gamma/size 
        elif adapt_gamma == "OldBalanced":
            if abs(sum_norm_ui_sq)>1e-20:
                gamma =  np.sqrt(size*norm_sumy_sq/sum_norm_ui_sq) # this is the original version
        elif adapt_gamma == "NewBalanced":
            if abs(sum_norm_ui_sq)>1e-20:
                gamma =  (size*norm_sumy_sq/sum_norm_ui_sq) # new version using eta conversion.


        normGrad = gamma**(-1)*norm_sumy_sq + sum_norm_ui_sq


        if abs(normGrad)>1e-20:
            z = z - (gamma**(-1)*phi/normGrad)*sumy
            wi = wi - (phi/normGrad)*ui

        #print("P"+str(i)+" norm z = "+str(np.linalg.norm(z))+", norm wi = "+str(np.linalg.norm(wi)))


        if i == 0:
            normGrads.append(normGrad)
            phis.append(phi[0])
            func_vals.append(lasso_val(z,A,b,lam))
            ratio_gradz2w.append(norm_sumy_sq/sum_norm_ui_sq)
            #phis_after.append(phi_after)

    #if i == 0:
        #print("P"+str(i)+": z norm = "+str( np.linalg.norm(z) ))
        #print("P"+str(i)+": (lam /z norm)^2  = "+str( (lam/np.linalg.norm(z))**2 ))
    #print("P"+str(i)+": wi norm  = "+str( np.linalg.norm(wi) ))
    print("-------------------------------------------")
    print("P"+str(i)+": (wi norm /z norm)^2 = "+str( (np.linalg.norm(wi)/np.linalg.norm(z))**2 ))

    if i == 0:
        nnz = sum(abs(z)>1e-10)
        print("z nnz = "+str(nnz))
        print("final function value "+str(func_vals[-1]))

        if doPlots:
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
    return [z,func_vals[-1]]


def resid_balance(gamma,gradz_sq,gradw_sq,n):
    mu = 10.0
    tau = 2.0
    maxGamma = 1e6
    minGamma = 1e-6
    #print("gradw^2/n = "+str(gradw_sq/n))
    #print("gradz^2 = "+str(gradz_sq))
    if (gradw_sq/n >= mu*gradz_sq):
        gamma = min(maxGamma,gamma*tau)
    elif(mu*gradw_sq/n <= gradz_sq):
        gamma = max(minGamma,gamma/tau)

    return gamma

def lasso_val(z,A,b,lam):
    return 0.5*np.linalg.norm(A.dot(z)-b,2)**2 + lam*np.linalg.norm(z,1)

def proxL1(a,thresh):
    x = (a> thresh)*(a-thresh)
    x+= (a<-thresh)*(a+thresh)
    return x

def update_block(z,wi,rho,Ai,bi,lam,Delta):
    gradz = Ai.T.dot(Ai.dot(z) - bi)
    continueBT = True
    while continueBT:
        t = z - rho*(gradz - wi)
        x = proxL1(t,rho*lam)
        a = (1/rho)*(t-x)
        gradx = Ai.T.dot(Ai.dot(x) - bi)
        y = a + gradx
        left = Delta*np.linalg.norm(z - x,2)**2
        right = (z-x).T.dot(y - wi)
        if left <= right:
            # backtrack terminates
            continueBT = False
        else:
            rho = rho*0.7
    return [x,y,rho]



