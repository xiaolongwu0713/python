'''
This script first extract the decoding result (correlation coefficient (CC)) outputted by channel_removal.py.
Then, it will plot the ccs.
It shows how CC changes with the removal of the most informative channel.
'''

from speech_Ruijin.config import *
from speech_Ruijin.baseline_linear_regression.opt import opt
import re
import matplotlib.pyplot as plt

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
for result in results:
    ax.plot(result)

legend=['sid'+str(sid) for sid in sids]
ax.legend(legend)
figname=result_file+'channel_removal.pdf'
fig.savefig(figname)










