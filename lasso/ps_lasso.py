import numpy as np




def create_simple_partition_v0(n,nparts):
    # Note that this version is incompatable with python3! due to the 
    # extend method no longer being defined by range()
    partsize = int(n/nparts)
    partition = [range(i*partsize,(i+1)*partsize) for i in range(nparts)]
    if n%nparts != 0:
        partition[-1].extend(range(nparts*partsize,n))
    return partition

def create_simple_partition(n,nparts):    
    # simplest way to deal with the situation where nparts does not dive n.
    # Just add the remainder to the final slice. This is not ideal for 
    # breaking up work for processors as one single processor gets more work
    # than the others... So try to set num processors as a factor of n. 
    partsize = int(n/nparts)           
    partition = [[i*partsize,(i+1)*partsize] for i in range(nparts)]
    if n%nparts != 0:
        partition[-1][1] = n
        
    
    return partition




def estimate_pd_scale(A,b,lam,p):
   '''
    We want to know ||w_i^*||/||z^*|| for the solution point (z^*,w^*).
    Here, we estimate this by randomly generating p points and computing
    ||g||/||z|| for g a randomly selected subgradient.
   '''
   m,d = A.shape
   z = np.random.normal(0,1,[p,d])
   ratios_sq = 0.0
   for i in range(p):
       sgs = get_sg(z[i],A,b,lam)
       ratios_sq += (sgs/np.linalg.norm(z[i]))**2
   av_ratio_sq = ratios_sq/p
   return av_ratio_sq


def get_sg(x,A,b,lam):
    return np.linalg.norm(A.T.dot(A.dot(x)-b)+lam*SGN(x),2)


def SGN(x):
    '''
    This is an arbitrary subgradient of the ell1 norm where for any zero sum_entries
    we just return a random number uniform between -1 and 1.
    '''
    d = len(x)

    return (1.0*(x>0)-1.0*(x<0) + 1.0*(x==0)*(2.0*np.random.random(d)-1.0) )
