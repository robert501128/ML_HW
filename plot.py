import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
mu=0
var=1
sigma=math.sqrt(var)
x=np.linspace(-4,4,1000)

mu2=9.5/6
var2=1/6.0
sigma2=math.sqrt(var2)

mu3=1.625
var3=1/8.0
sigma3=math.sqrt(var3)
plt.plot(x,mlab.normpdf(x,mu,sigma))
plt.plot(x,mlab.normpdf(x,mu2,sigma2))
plt.plot(x,mlab.normpdf(x,mu3,sigma3))

plt.show()