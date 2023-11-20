import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy

from gesture.config import *
from gesture.utils import read_good_sids

sids=read_good_sids()
### test the confusion matrix plot ###
fig,ax=plt.subplots()
a=np.asarray([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.2,0.3,0.9]])
ax.imshow(a)


### test violin plot ####
sids=[4, 10, 13, 17, 18, 29, 32, 41]
accs=[]
# original acc
accs.append(
    np.asarray([
    [.73,.79,.74,.67,.75],
    [.90,.93,.90,.86,.88],
    [.81,.83,.73,.80,.85],
    [.82,.83,.87,.82,.78],
    [.57,.53,.52,.52,.48],
    [.68,.57,.66,.68,.65],
    [.62,.58,.56,.57,.52],
    [.99,.99,1,.99,.99]
    ])
)
# NI acc
accs.append(
    np.asarray([
    [.78,.76,.73,.67,.73],
    [.92,.90,.91,.88,.84],
    [.83,.87,.78,.82,.82],
    [.81,.82,.85,.87,.77],
    [.58,.49,.45,.51,.54],
    [.66,.65,.57,.64,.68],
    [.55,.54,.59,.55,.49],
    [.99,.99,.99,1,.98]
    ])
)
# VAE acc
accs.append(
    np.asarray([
    [.79,.77,.76,.70,.73],
    [.94,.91,.91,.90,.86],
    [.81,.75,.79,.82,.85],
    [.85,.79,.78,.86,.82],
    [.58,.56,.54,.54,.47],
    [.66,.62,.69,.68,.76],
    [.59,.56,.53,.58,.63],
    [.98,.99,.99,1,.99]
    ])
)

# CWGANGP acc
accs.append(
    np.asarray([
    [.77,.79,.75,.72,.82],
    [.91,.92,.94,.90,.86],
    [.80,.87,.77,.88,.85],
    [.83,.84,.79,.85,.89],
    [.58,.58,.56,.50,.60],
    [.72,.68,.77,.64,.70],
    [.63,.61,.65,.59,.56],
    [1,1,.99,1,.98]
    ])
)
# TTSCGAN acc
accs.append(
    np.asarray([
    [.83,.85,.72,.78,.81],
    [.87,.94,.94,.95,.94],
    [.85,.78,.83,.86,.89],
    [.86,.84,.87,.80,.92],
    [.58,.64,.59,.54,.64],
    [.70,.66,.68,.76,.76],
    [.68,.64,.65,.63,.78],
    [1,1,.98,1,.99]
    ])
)

## write to csv file, import to SPSS and perform a two-way analysis
methods=['orig','NI','VAE','CWGANGP','CTTSCGAN']
# Item: compare only two method
compare_accs=[]
compare_methods=[]
compare_accs.append(accs[3])
compare_accs.append(accs[4]) # only compare methods 3 and 4
compare_methods.append(methods[3])
compare_methods.append(methods[4])
# Item: compare all method
compare_accs=accs
compare_methods=methods

acc_csv=[]
for i,alg in enumerate(compare_methods):
    for j, p in enumerate(sids):
        for k in range(5): # 5-fold
            # 5 accs from 5-fold
            tmp=[]
            tmp.append(compare_accs[i][j, k])
            tmp.insert(0, j + 1.0)
            tmp.insert(0, i + 1.0)
            acc_csv.append(tmp)
#acc_csv=np.asarray([a.reshape(40).tolist() for a in accs]).transpose().tolist()
filename=tmp_dir+'del1.csv'
if os.path.exists(filename):
    os.remove(filename)
with open(filename, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    #writer.writerow(methods)
    writer.writerows(acc_csv)


save_dir=result_dir + 'DA/plots/'
fig,ax=plt.subplots()
ax.clear()
for i, acc in enumerate(range(len(accs))):
    positions = [2+1.3*i for i in range(8)] # [2, 3.5, 5, 8, 10, 12, 14, 16]
    positions=[j+i*0.2 for j in positions]
    violin_parts = ax.violinplot(accs[i].transpose(),widths=0.15, positions=positions,showmeans=True,showextrema = True,)
    tab_colors_names=list(mcolors.TABLEAU_COLORS.keys())
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = violin_parts[partname]
        vp.set_linewidth(0.5)

    for vp in violin_parts['bodies']:
        vp.set_facecolor(tab_colors_names[i])
        vp.set_edgecolor(tab_colors_names[i])
        vp.set_linewidth(0.0)
        vp.set_alpha(0.5)
import matplotlib.colors as mcolors
#RGBcode=[mcolors.to_rgb(tab_colors_codes[i]) for i in range(len(accs))]
#ax.legend(['orig','NI','VAE','CWGANGP','CTTSCGAN'],facecolor=RGBcode,loc="lower left",bbox_to_anchor=(0,1,0.1,0.1),fontsize='small')

positions = [2+1.3*i for i in range(8)] # [2, 3.5, 5, 8, 10, 12, 14, 16]
positions=[j+2*0.2 for j in positions]
ax.set_xlim([1.5,12.5])
ax.set_xticks(positions) #指定要标记的坐标
ticklabels=['S'+str(i) for i in list(range(1,accs[0].shape[0]+1))]
ax.set_xticklabels(ticklabels,rotation = 0, position=(0,0),fontsize=10)
ax.set_ylabel('Decoding Accuracy', fontsize=10, labelpad=5)
filename = save_dir + 'compare_DA_acc_violin_data.pdf' # add the data plot to the coordinator
fig.savefig(filename)


#### model selection according to the training process ####
from tbparse import SummaryReader
from pre_all import *
event_dir='H:/Long/data/gesture/DA/TTSCGAN/3/2022_11_18_19_41_49/Log/'
#
file=event_dir+'monitor_r_acc_adv/events.out.tfevents.1668771710.workstation.45308.6'
reader = SummaryReader(file) # long format
df = reader.scalars
r_acc_adv=df.iloc[:,2]
#
file=event_dir+'monitor_f_acc_adv/events.out.tfevents.1668771710.workstation.45308.7'
reader = SummaryReader(file) # long format
df = reader.scalars
f_acc_adv=df.iloc[:,2]
#
file=event_dir+'monitor_r_acc_cls/events.out.tfevents.1668771710.workstation.45308.8'
reader = SummaryReader(file) # long format
df = reader.scalars
r_acc_cls=df.iloc[:,2]
#
file=event_dir+'monitor_f_acc_cls/events.out.tfevents.1668771710.workstation.45308.9'
reader = SummaryReader(file) # long format
df = reader.scalars
f_acc_cls=df.iloc[:,2]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

all_events=[r_acc_adv,f_acc_adv,r_acc_cls,f_acc_cls]
fig,ax=plt.subplots()
ax.clear()
for i in range(len(all_events)):
    # original event data
    #ax.plot(all_events[i],linewidth=0.2,alpha=0.1,color=tab_colors_codes[i])
    # smoothing the data
    ax.plot(smooth(all_events[i].to_numpy(),100),linewidth=0.5,alpha=1,color=tab_colors_codes[i])
ax.legend(['r_acc_adv','f_acc_adv','r_acc_cls','f_acc_cls'],fontsize=10)
filename=result_dir+'DA/plots/training_process_standard_loss.pdf'
fig.savefig(filename)

## d_loss reach to 0 using WGANGP method ###
event_dir='H:/Long/data/gesture/DA/CWGANGP/10/2022_11_23_11_42_26/events.out.tfevents.1669174950.workstation.2236.0'
reader = SummaryReader(event_dir) # long format
df = reader.scalars
d_loss=df[df['tag']=='d_loss']['value'].to_numpy()
ax.clear()
ax.plot(smooth(d_loss,100),linewidth=0.7)
filename=result_dir+'DA/plots/training_process_wgangp_d_loss.pdf'
fig.savefig(filename)


#### stat analysis ####
orig=list(accs[0].reshape(1,-1))[0]
NI=list(accs[1].reshape(1,-1))[0]
VAE=list(accs[2].reshape(1,-1))[0]
CWGANGP=list(accs[3].reshape(1,-1))[0]
CTTSCGAN=list(accs[4].reshape(1,-1))[0]

# Wilcoxon signed-rank test between x and y: scipy.stats.wilcoxon(x,y)
res = scipy.stats.wilcoxon(CWGANGP,CTTSCGAN)
res.pvalue
#### stat analysis per subject ####
sub_accs={}
for s,sid in enumerate(sids):
    key = 'sid' + str(s)
    sub_accs[key] = {}
    for m, method in enumerate(methods):
        sub_accs[key][method]=accs[m][s,:]
sub_acc_stat={}
for s,sid in enumerate(sids):
    key = 'sid' + str(s)
    sub_acc_stat[key] = {}
    for m, method in enumerate(methods):
        sub_acc_stat[key][method]=[]
        sub_acc_stat[key][method].append(sub_accs[key][method].mean())
        sub_acc_stat[key][method].append(sub_accs[key][method].std())
    sid_acc=sub_accs[key]
    res = scipy.stats.wilcoxon(sid_acc[list(sid_acc)[-1]], sid_acc[list(sid_acc)[-2]])
    sub_acc_stat[key]['pvale']=res.pvalue

## get mean and std for sidx and methodx: sub_acc_stat[stdx][methodx]
## then make a table









