
import numpy as np

n = 1000
d = 1000
sigma = 1.0/np.sqrt(n)
A = np.random.normal(0,sigma,[n,d])
b = np.random.normal(0,sigma,n)

np.save('A.npy',A)
np.save('b.npy',b)
