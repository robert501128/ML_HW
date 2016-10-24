import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random 
s=900
data=10000
def lik(p):
    w = p[:-1].reshape((4,1))
    sigma = p[-1]
    _x=np.array([x**0,x,x**2,x**3])
    y=sum(w*_x)
    L = 1/(2*sigma**2)*sum((y-t)**2) +\
     len(x)/2.0*(np.log(sigma**2))+len(x)/2.0*np.log(2*np.pi)
    return L

random.seed(s)
x = np.array([random.uniform(-100,100) for i in range(data)])
x.sort()
np.random.seed(s)
t= np.random.normal(0.1+2*x+x**2+3*x**3,1)
lik_model = minimize(lik, np.array([1.0,1.0,1.0,1.0,1.0]), method='Powell')
w = lik_model['x'][:-1].reshape((4,1))
_x=np.array([x**0,x,x**2,x**3])
y=sum(w*_x) 
plt.scatter(x,t)
x = np.array([random.uniform(-100,100) for i in range(10000)])
x.sort()
_x=np.array([x**0,x,x**2,x**3])
y=sum(w*_x)
plt.plot(x,y)
plt.show()
