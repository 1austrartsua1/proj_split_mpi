from mpi4py import MPI
import numpy as np
import ps_lasso as ls
from matplotlib import pyplot as plt


def ps_mpi_sync_lasso(iter,A,b,lam,rho,gamma,Delta):
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

    print("Process "+str(i)+" of "+str(size)+". Norm of A = "+str(np.linalg.norm(A)))

    # each processor has its own local redundant copy of z
    z = np.zeros(d)

    # each processor has its own local (xi,yi) and wi
    wi = np.zeros(d)
    xi = np.zeros(d)
    yi = np.zeros(d)

    # processor 0 will plot results for us
    if i == 0:
        normGrads = []
        phis = []

    for k in range(iter):
        if i == 0:
            if k%100 == 0:
                print("iter "+str(k))

        # block update
        [xi,yi,rho] = update_block(z,wi,rho,A[partition[i]],b[partition[i]],lam/size,Delta)

        # projection updates
        phi_i = (z - xi).dot(yi - wi)
        phi = Comm.allreduce(phi_i) # scalar all reduce
        sumy = Comm.allreduce(yi)   # vector all reduce
        xav = (1.0/size)*Comm.allreduce(xi) # vector all reduce
        ui = xi - xav

        norm_ui_sq = np.linalg.norm(ui,2)**2
        sum_norm_ui_sq = Comm.allreduce(norm_ui_sq) # scalar all reduce

        normGrad = gamma**(-1)*np.linalg.norm(sumy,2)**2 + sum_norm_ui_sq

        z = z - (gamma**(-1)*phi/normGrad)*sumy

        wi = wi - (phi/normGrad)*ui

        #phi_i = (z - xi).dot(yi - wi)
        #phi_after = Comm.allreduce(phi_i)

        if i == 0:
            normGrads.append(normGrad)
            phis.append(phi)
            #phis_after.append(phi_after)
        #print("P"+str(i)+": z shape = "+str(z.shape))
        #print("P"+str(i)+": wi shape = "+str(wi.shape))

    if i == 0:
        plt.semilogy(normGrads)
        plt.title("norm of gradients of phi")
        plt.show()
        plt.semilogy(phis)
        plt.title("phis (should be positive)")
        plt.show()

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



    return [z,z,1.0]
