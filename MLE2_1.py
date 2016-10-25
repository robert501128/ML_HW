import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random 
s=900
data=100
def lik(p):
    w = p[:-1].reshape((4,1))
    sigma = p[-1]
    _x=np.array([x**0,x,x**2,x**3])
    y=sum(w*_x)
    L = 1/(2*sigma**2)*sum((y-t)**2) +\
     len(x)/2.0*(np.log(sigma**2))+len(x)/2.0*np.log(2*np.pi)
    return L

f, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', sharey='row')

random.seed(s)
x = np.array([random.uniform(-100,100) for i in range(data)])
x.sort()
np.random.seed(s)
t= np.random.normal(0.1+2*x+x**2+3*x**3,1)
lik_model = minimize(lik, np.array([1.0,1.0,1.0,1.0,1.0]), method='Powell')
w = lik_model['x'][:-1].reshape((4,1))
fig= plt.figure() 

ax1.scatter(x,t,label='t')
x = np.array([random.uniform(-100,100) for i in range(10000)])
x.sort()
_x=np.array([x**0,x,x**2,x**3])
y=sum(w*_x)


ax1.plot(x,y,label='y')
ax1.set_title('N=100')
strGraph=""
for i in range(len(w)):
	strGraph+="w"+str(i)+": %.10f\n" % w[i][0]
strGraph+='beta: %.10f' % lik_model['x'][-1]
ax1.text(0.95, 0.01, strGraph,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='blue', fontsize=10)

data=1000
random.seed(s)
x = np.array([random.uniform(-100,100) for i in range(data)])
x.sort()
np.random.seed(s)
t= np.random.normal(0.1+2*x+x**2+3*x**3,1)
lik_model = minimize(lik, np.array([1.0,1.0,1.0,1.0,1.0]), method='Powell')
w = lik_model['x'][:-1].reshape((4,1))
fig= plt.figure() 
ax2.scatter(x,t,label='t')
x = np.array([random.uniform(-100,100) for i in range(10000)])
x.sort()
_x=np.array([x**0,x,x**2,x**3])
y=sum(w*_x)


ax2.plot(x,y,label='y')
ax2.set_title('N=1000')
strGraph=""
for i in range(len(w)):
	strGraph+="w"+str(i)+": %.10f\n" % w[i][0]
strGraph+='beta: %.10f' % lik_model['x'][-1]
ax2.text(0.95, 0.01, strGraph,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes,
        color='blue', fontsize=10)

data=10000
random.seed(s)
x = np.array([random.uniform(-100,100) for i in range(data)])
x.sort()
np.random.seed(s)
t= np.random.normal(0.1+2*x+x**2+3*x**3,1)
lik_model = minimize(lik, np.array([1.0,1.0,1.0,1.0,1.0]), method='Powell')
w = lik_model['x'][:-1].reshape((4,1))
fig= plt.figure() 
ax3.scatter(x,t,label='t')
x = np.array([random.uniform(-100,100) for i in range(10000)])
x.sort()
_x=np.array([x**0,x,x**2,x**3])
y=sum(w*_x)


ax3.plot(x,y,label='y')
ax3.set_title('N=10000')
strGraph=""
for i in range(len(w)):
	strGraph+="w"+str(i)+": %.10f\n" % w[i][0]
strGraph+='beta: %.10f' % lik_model['x'][-1]
ax3.text(0.95, 0.01, strGraph,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax3.transAxes,
        color='blue', fontsize=10)
ax1.axes.get_yaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax3.axes.get_yaxis().set_visible(False)




plt.show()
