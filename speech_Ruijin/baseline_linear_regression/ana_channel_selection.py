'''
This script first extract the decoding result (correlation coefficient (CC)) outputted by channel_selection.py.
Then, it will plot the result.
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

result_file = data_dir + 'baseline_LR_channel_selection/SingleWordProductionDutch/mel_' + str(melbins)+'/'
corrs=[]
opt_channels=[]
error_sids=[]
for sid in sids:
    filei=result_file+ 'sid' + str(sid) + '/'+ 'result.txt'
    f = open(filei, 'r')
    completed=False
    for line in f:
        astring=line.strip()
        if astring.startswith('Best corrs'):
            tmp=astring.split(':')[1]
            corr=[float(i) for i in tmp.split(',') if i not in [',','.']]
            completed=True
        if astring.startswith('Final opt channel'):
            tmp=astring.split(':')[1]
            opt_channel=[int(i) for i in tmp.split(',') if i not in [',','.']] #float(re.findall('Final opt channel:(.*).', astring)[0])
    if completed==False:
        error_sids.append(sid)
    corrs.append(corr)
    opt_channels.append(opt_channel)

if len(error_sids)>1:
    raise ValueError('Selection program didn\'t finish correctly on sid:'+','.join(str(sid) for sid in error_sids)+'.')

fig,ax=plt.subplots()
for tmp in corrs:
    ax.plot(tmp)

legend=['sid'+str(sid) for sid in sids]
ax.legend(legend)
figname=result_file+'channel_selection.pdf'
fig.savefig(figname)










