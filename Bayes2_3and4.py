import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random 
s=900
alpha=100000000000000000000
beta=1.0
BayesS_1=None
plotSize=100
def BayesM(inp):
    s=np.zeros([6,1])
    for i in range(len(x)):
        _x=np.array([[x[i]**0,x[i]**1,x[i]**2,x[i]**3,x[i]**4,x[i]**5]])
        s+=_x.T*t[i]
    _inp=np.array([[inp**0,inp**1,inp**2,inp**3,inp**4,inp**5]])
    return beta*np.dot(np.dot(_inp,BayesS_1),s)
def BayesS2(inp):
    _inp=np.array([[inp**0,inp**1,inp**2,inp**3,inp**4,inp**5]])
    ans=1/beta+np.dot(np.dot(_inp,BayesS_1),_inp.T)
    return ans


ax=[None for i in range(4)]
f, ((ax[0], ax[1]),(ax[2], ax[3])) = plt.subplots(2,2, sharex='col')
data=[10,10,10,10]

for i in range(4):
    random.seed(s)
    x = np.array([random.uniform(-100,100) for j in range(data[i])])
    x.sort()
    np.random.seed(s)
    t= np.random.normal(0.1+2*x+x**2+3*x**3,1/beta)
    sXX=np.zeros([6,6])
    for j in range(data[i]):
        _x=np.array([[x[j]**0,x[j],x[j]**2,x[j]**3,x[j]**4,x[j]**5]])
        sXX+=np.dot(_x.T,_x)
    BayesS_1=inv(alpha*np.identity(6)+beta*sXX)
    
    fig= plt.figure() 
    ax[i].scatter(x,t)
    px = np.array([random.uniform(-100,100) for j in range(plotSize)])
    px.sort()
    y=np.zeros(plotSize)
    var=np.zeros(plotSize)
    for j in range(len(px)):
        y[j]=BayesM(px[j])
        var[j]=BayesS2(px[j])
    ax[i].plot(px,y)
    ax[i].set_title('N='+str(data[i]))
    ax[i].fill_between(px, y+var, y-var, alpha=.5)
    ax[i].axes.get_yaxis().set_visible(False)

plt.show()