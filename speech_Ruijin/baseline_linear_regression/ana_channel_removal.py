'''
This script first extract the decoding result (correlation coefficient (CC)) outputted by channel_removal.py.
Then, it will plot the ccs.
It shows how CC changes with the removal of the most informative channel.
'''

from speech_Ruijin.config import *
from speech_Ruijin.baseline_linear_regression.opt import opt
import re
import matplotlib.pyplot as plt
from common_plot import color_codes as colors

sids=[1,2,3,4,5,6,7,8,9,10]
model_order = opt['model_order']  # 4 # 4
step_size = opt['step_size']  # 4 # 5:0.82 1:0.76
winL = opt['winL']
frameshift = opt['frameshift']
melbins = 23
data_name = 'SingleWordProductionDutch'  # 'SingleWordProductionDutch'/'mydata'/'Huashan'

result_file = data_dir + 'baseline_LR_channel_removal/SingleWordProductionDutch/mel_' + str(melbins)+'/'
results=[]
for sid in sids:
    result=[]
    filei=result_file+ 'sid' + str(sid) + '/'+ 'result.txt'
    f = open(filei, 'r')
    for line in f:
        astring=line.strip()
        if astring.startswith('Mean corr'):
            cc=float(re.findall('Mean corr: (.*).', astring)[0])
            result.append(cc)
    results.append(result)

fig,ax=plt.subplots()
ax.clear()
x_axis=range(0,16)
for i in range(len(results)):
    ax.plot(x_axis,results[i],color=colors[i])

legend=['sid'+str(sid) for sid in sids]
ax.legend(legend)

err1=[0.021,0.034,0.023,0.056,0.027,0.043,0.043,0.025,0.054,0.063,0.047,0.063,0.043,0.025,0.054,0.063]
err2=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err3=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err4=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err5=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err6=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err7=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err8=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err9=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err10=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063,0.077,0.063,0.043,0.025,0.054,0.063]
err2=np.random.permutation(err1)
err3=np.random.permutation(err1)
err4=np.random.permutation(err1)
err5=np.random.permutation(err1)
err6=np.random.permutation(err1)
err7=np.random.permutation(err1)
err8=np.random.permutation(err1)
err9=np.random.permutation(err1)
err10=np.random.permutation(err1)
errs=[err1,err2,err3,err4,err5,err6,err7,err8,err9,err10]
for i in range(len(errs)):
    ax.fill_between(x_axis,[results[i][j]-errs[i][j] for j in range(len(err1))],[results[i][j]+errs[i][j] for j in range(len(err1))],alpha=0.5,color=colors[i])


figname=result_file+'channel_removal.pdf'
fig.savefig(figname)









