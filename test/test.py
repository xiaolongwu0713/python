import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

fig,ax=plt.subplots()
xdata = np.linspace(1, 19, 19)
y1=np.asarray([float(i) for i in [1,1,2,3,4,9,5,3,11,9,8,3,9,6,8,7,6,5,1]])
y2=np.asarray([1,0,1,1,0,1,1,1,2,3,2,1,3,4,3,4,2,4,3])
def func(x, m, a, b, c):
    return m*x**3+a*x**2+b*x+c

y=y1
popt, pcov = curve_fit(func, xdata, y)
ax.plot(xdata, func(xdata, *popt))
ax.clear()


a=[4,5,7,6,5,3,1,2,2,5,9,8,1,5,3,2,12,1,1,10,8,7,1,5,1,1,3,2,5,1,1,1,32,3,3,2,5,3,4,5,2,3,4,3,2,6,3,6,6,4,2,5,5,1,1,1,2,4,3,3,4,2,3,2,25,3,1,1]
v,c=np.unique(a, return_counts=True)
ax.bar(v,c)
fig.savefig('del.pdf')
