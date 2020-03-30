
import numpy as np

n = 50
d = 100
#sigma = 1.0/np.sqrt(n)
sigma = 1.0
factor = 1.0
sigma *= factor
type = 'normal'
if type == 'normal':
    A = np.random.normal(0,sigma,[n,d])
elif type == 'pm1':
    A = 2*(np.random.normal(0,sigma,[n,d])>0)-1.0

b = np.random.normal(0,sigma,n)

np.save('A.npy',A)
np.save('b.npy',b)
np.savetxt('A.txt',A)
np.savetxt('b.txt',b)

