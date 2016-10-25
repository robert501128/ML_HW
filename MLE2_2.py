import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random 
s=900

def lik(p):
    w = p[:-1].reshape((6,1))
    sigma = p[-1]
    _x=np.array([x**0,x,x**2,x**3,x**4,x**5])
    y=sum(w*_x)
    L = 1/(2*sigma**2)*sum((y-t)**2) +\
     len(x)/2.0*(np.log(sigma**2))+len(x)/2.0*np.log(2*np.pi)
    return L

ax=[None for i in range(4)]
f, ((ax[0], ax[1]),(ax[2], ax[3])) = plt.subplots(2,2, sharex='col')
data=[10,100,1000,10000]

for i in range(4):
	random.seed(s)
	x = np.array([random.uniform(-100,100) for j in range(data[i])])
	x.sort()
	np.random.seed(s)
	t= np.random.normal(0.1+2*x+x**2+3*x**3,1)
	lik_model = minimize(lik, np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0])\
		, method='Powell')
	w = lik_model['x'][:-1].reshape((6,1))

	fig= plt.figure() 
	ax[i].scatter(x,t,label='t')
	x = np.array([random.uniform(-100,100) for j in range(10000)])
	x.sort()
	_x=np.array([x**0,x,x**2,x**3,x**4,x**5])
	y=sum(w*_x)
	ax[i].plot(x,y)
	ax[i].set_title('N='+str(data[i]))
	ax[i].axes.get_yaxis().set_visible(False)
	strGraph=""
	for j in range(len(w)):
		if j%2==0:
			strGraph+="w"+str(j)+": %.10f " % w[j][0]
		else:
			strGraph+="w"+str(j)+": %.10f\n" % w[j][0]
	strGraph+='beta: %.10f' % lik_model['x'][-1]
	print strGraph
	
plt.show()

