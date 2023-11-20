# Test some plot function
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import *

# Creating dataset
np.random.seed(10)
acc_reference = [0.56, 0.48, 0.67,0.85, 0.84, 0.52, 0.43,0.67, 0.52,0.99]
acc = [0.55, 0.40, 0.61,0.81, 0.8, 0.50, 0.42,0.61, 0.54,0.97]
data=[acc_reference, acc]

fig,ax = plt.subplots()
ax.plot(np.ones(len(acc_reference)),acc_reference,'ro',markersize=12, alpha=0.2)
ax.plot(np.ones(len(acc))*2,acc,'bo',markersize=12, alpha=0.2)
ax.legend(['Accu_reference','Acc_no_reference'],loc="lower left",bbox_to_anchor=(0,1,0.1,0.1),fontsize='small')
ax.boxplot(data)

filename='/Users/long/My Drive/python/gesture/result/dl_result/acc_with_reference.pdf'
plt.savefig(filename)

from scipy.stats import mannwhitneyu
U1, p = mannwhitneyu(acc, acc_reference, method="exact")
p


windsize=[200, 400,600,800,1000,1200] # 400 best
acc_var_wind=[0.55, 0.77,0.75,0.68,0.6,0.5] #0.77
stepsize=[20,50,100,200,300,400,500] # 100 best
acc_var_step=[0.81,0.80,0.85,0.84,0.76,0.77,0.73] #0.77
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(windsize,acc_var_wind,'bo',markersize=12, alpha=0.2)
ax1.set_xticks(windsize)
ax1.set_xticklabels(windsize)
ax1.tick_params(axis='x', labelcolor= 'tab:blue')
ax1.legend(['window size'],loc="lower left",bbox_to_anchor=(0,0.5,0.1,0.1),fontsize='small')
ax1.set_ylabel('Decoding accuracy', fontsize=10, labelpad=5)

ax2.plot(stepsize,acc_var_step,'ro',markersize=12, alpha=0.2)
ax2.set_xticks(stepsize)
ax2.set_xticklabels(stepsize)
ax2.tick_params(axis='x', labelcolor= 'tab:red')
ax2.legend(['Slide stride'],loc="lower left",bbox_to_anchor=(0,0.6,0.1,0.1),fontsize='small')

filename='/Users/long/My Drive/python/gesture/result/dl_result/acc_with_wind_sride.pdf'
plt.savefig(filename)




