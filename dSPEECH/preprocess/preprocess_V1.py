'''
Paradigm Version 1;
compare triggers from matlab logfile and Natus;
Hospital laptop is about 10 seconds behind (Needs to be verified);
Subjects: SEEG1,2;
'''
import calendar
import datetime
import glob
from util.util_MNE import delete_annotation, keep_annotation
from dSPEECH.config import *
plott=False

sf=1024
type='SEEG' #'SEEG/ECoG
sid=2 # SEEG 1/2
# get the channel names from the recon.ppt file
channel_names=['TP','Am','HA','HP','BT1','BT2','OF','FO','ST1','PO','ST2','ST3','SM','LHA']
eeg_file=data_dir+'raw/'+type+str(sid)+'_*/EEG.edf' # SEEG1_datetime
eeg_file=os.path.normpath(glob.glob(eeg_file)[0])
raw = mne.io.read_raw_edf(eeg_file,preload=True)
# set the channel types
raw.info.get_channel_types() # All chs are type of 'eeg'
ch_types={}
for chi in raw.ch_names:
    if any([tmp in chi for tmp in channel_names]):
        ch_types[chi]='eeg'
    else:
        ch_types[chi] = 'misc'
raw.set_channel_types(ch_types)
raw.pick(['eeg'])

## remove non-eeg channels + deal with possible low frequency drift and line noise
# SEEG2 exhibit strong line noise;
# SEEG2 identify additional bad channels: ST3-5
if plott: raw.plot_psd(tmin=0, tmax=600,average=False)
raw.info["bads"].append("ST3-5") #raw.info["bads"].remove("PR")
raw.pick(picks='all',exclude='bads')
#raw.pick(exclude='bads')
# PSD again
if plott: raw.plot_psd(tmin=0, tmax=600,average=False,exclude="bads")
# line noise
freqs = (50,100,150,200,250,300,350,400,450)
raw.notch_filter(freqs=freqs)
# possible low frequency drift
cutoff=0.1
raw.filter(l_freq=cutoff, h_freq=None)

## working with triggers
plot_scale=100e-5
plot_channels=5
if plott: raw.plot(duration=100,time_format='clock',n_channels=plot_channels, scalings=dict(eeg=50e-5))
##  extract annotations='TRIG[001]:1'
anno_natus=raw.annotations
if type=='ECoG':
    event_descripion='Patient Event'
else:
    event_descripion = 'TRIG[001]:1'
anno_natus=keep_annotation(anno_natus,event_descripion)
raw.set_annotations(anno_natus)
if plott: raw.plot(duration=100,n_channels=plot_channels,scalings=dict(eeg=50e-5))

## 1, export prompts to inf.txt from matlab using export_logfile_to_text.m file;
# 2, (Fixed in version 2) Manually add time stamp to 'Resumeing' event. The time of resuming is the same as that of next sentence;
# 2.5, Save the file;
# 3, Read inf.txt, and format it to lines in the form of "time, sentence";
filename=data_dir+'raw/'+type+str(sid)+'_*/matlab/inf.txt'
filename=os.path.normpath(glob.glob(filename)[0])
#filename=r'D:\data\speech_Southmead\raw\SEEG1\matlab\inf.txt'
with open(filename,"r") as infile:
    lines_tmp=infile.read().split('\n')
prompts_tmp=[tmpi for tmpi in lines_tmp if len(tmpi)>0] #

# format to lines in the form of "time, sentence";
prompts=[] # 106
for i in range(int(len(prompts_tmp)/2)):
    tmp=[]
    tmp.append(prompts_tmp[i*2])
    tmp.append(prompts_tmp[i*2+1])
    prompts.append(tmp)

## create matlab annotation from the prompts : append(onset, duration, description, ch_names=None)
callider_dict={month: index for index, month in enumerate(calendar.month_abbr) if month}
orig_time=anno_natus[0]['orig_time']#datetime.datetime(2023, 11, 17, 9, 51, 34, tzinfo=datetime.timezone.utc)
onsets=[]
durations=[]
descriptions=[]
for prompt in prompts:
    time=prompt[0]
    year=int(time.split(' ')[0].split('-')[-1])
    month_tmp = time.split(' ')[0].split('-')[1]
    month=callider_dict[month_tmp]
    day = int(time.split(' ')[0].split('-')[0])
    hour = int(time.split(' ')[1].split(':')[0])
    min = int(time.split(' ')[1].split(':')[1])
    sec = int(time.split(' ')[1].split(':')[2])
    ts=datetime.datetime(year, month, day, hour, min, sec, tzinfo=datetime.timezone.utc)

    onset=(ts-orig_time).total_seconds()
    onsets.append(onset)
    durations.append(0)
    if prompt[1]=='Pausing':
        descriptions.append('pause')
    elif prompt[1]=='Resuming':
        descriptions.append('resume')
    elif prompt[1]=='Escape':
        descriptions.append('escape')
    else:
        descriptions.append('TRIG-matlab')
anno_matlab=mne.Annotations(onset=onsets,duration=durations,description=descriptions,orig_time=orig_time)# 106=102triggers+4pauses

## combine annotation from natus and matlab
ann_comb=anno_natus.__add__(anno_matlab)
raw1=raw.copy()
raw1.set_annotations(ann_comb)
if plott: raw1.plot(duration=100,n_channels=plot_channels,scalings=dict(eeg=50e-5))

## visually compare triggers from Natus and matlab;
# Add two annotations: program_start, program_end in annotation mode;
# (1,Press 'a' enter annotation mode; 2, type program_start, then hit enter; 3, click and drag will create program_start point;
# 4, type program_end, then hit enter; 5, click and drag will create program_end point;)
# The raw1.annotation will reflect the modified annotation in real-time
# (no need to close the figure);

# manually add program_start and program_end annotations
if plott:
    raw1.plot(duration=100,n_channels=plot_channels,scalings=dict(eeg=50e-5))
else:
    onsets = [356, 3158]  # s
    durations = [5, 5]  # s
    descriptions = ['program_start', 'program_end']
    anno_start_end = mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=orig_time)
    ann_comb = ann_comb.__add__(anno_start_end)
    raw1.set_annotations(ann_comb)
## extract the experimental data by only keeping program_start and program_end triggers;
seg_anno=raw1.annotations.copy()
seg_anno=keep_annotation(seg_anno, ['program_start','program_end'])
# get start time from 'program_start' trigger
tmin=seg_anno[0]['onset']-5 # extra 5 seconds
tmax=seg_anno[1]['onset']+seg_anno[1]['duration']+5 # extra 5 seconds
raw2=raw1.crop(tmin=tmin,tmax=tmax)
#raw2=raw1.crop_by_annotations(seg_anno) # this will only return data within each annotation period
if plott: raw2.plot(duration=100,time_format='float',n_channels=plot_channels,scalings=dict(eeg=50e-5)) # 'float'/'clock'

## SEEG1,SEEG2: replace the first 'TRIG[001]:1' with 'keyboard_wait'
# trigger sequence: 'program_start', 'TRIG[001]:1',.....,'program_end';
anno_tmp=raw2.annotations.copy()
replace_anno=mne.Annotations(onset=anno_tmp[1]['onset'],duration=anno_tmp[1]['duration'],description='keyboard_wait',orig_time=anno_tmp[1]['orig_time'])
anno_tmp.delete(1) # the second trigger('program_start', 'TRIG[001]:1',.....,'program_end')
ann_comb2=anno_tmp+replace_anno
raw2.set_annotations(ann_comb2)

## deal with the missing natus triggers
n_natus=[tmp for tmp in raw2.annotations if tmp['description']=='TRIG[001]:1' ]
n_matlab=[tmp for tmp in raw2.annotations if tmp['description']=='TRIG-matlab' ]
diff=len(n_matlab)-len(n_natus) # 4 triggers missing in n_natus
# visual check the missing natus triggers
if plott: raw2.plot(duration=100,time_format='float',n_channels=plot_channels,scalings=dict(eeg=50e-5)) # 'float'/'clock'
# First, estimate the time difference between triggers
raw2.annotations[20]['onset']-raw2.annotations[18]['onset'] # 15.25/15.125/15.11/15.13/15.125/15.127/15.132/15.135/15.132/
trigger_diff=15.125

# method1: annotate the range of missing triggers as new annotations with name 'missing1','missing2','missing3'.....
if plott:
    raw2.plot(duration=100,time_format='float',n_channels=plot_channels,scalings=dict(eeg=50e-5))
    insert=[]
    for m in range(diff):
        desc='missing'+str(m+1)
        missing_anno = [tmp for tmp in raw2.annotations if tmp['description'] == desc][0]
        for n in n_natus:
            if missing_anno['onset']<n['onset']+15<missing_anno['onset']+missing_anno['duration']:
                insert.append(n['onset'] + trigger_diff)  # s
else:
    # method2: indicate the time range of missing triggers
    #missings = [[321.051, 337.348], [753.128, 769.356], [784.389, 800.344], [1229.153, 1245.415]]
    missings=[[740,756],[860,877],[1598,1614],[2691,2708]] # SEEG2
    # calculate the missing onset
    insert = []
    for m in missings:
        for n in n_natus:
            if m[0] + tmin < n['onset'] + 15 < m[1] + tmin:
                insert.append(n['onset'] + trigger_diff)  # s
# create missing annotations
onsets=[]
durations=[]
descriptions=[]
for i in insert:
    onsets.append(i)
    durations.append(0)
    descriptions.append('TRIG[001]:1:inserted')
anno_missing=mne.Annotations(onset=onsets,duration=durations,description=descriptions,orig_time=orig_time)# 106=102triggers+4pauses
ann_comb2_added=ann_comb2.__add__(anno_missing)
raw2.set_annotations(ann_comb2_added)
#check result: contains annotations of 'program_start'+'keyboard_wait'+'TRIG[001]:1'+'TRIG[001]:1:inserted'+'pause'+'resume'+'program_end'


## delete trials right before 'pause' in annotation mode: select the annotation and right-click
if plott:
    raw2.plot(duration=100,time_format='float',n_channels=plot_channels,scalings=dict(eeg=50e-5)) # 'float'/'clock'
else:
    pass #TODO: implement the automaticl script (no manual annotation editing)
ann_comb2_added=raw2.annotations # SEEG2: 3 triggers deleted from ann_comb2_added
## delete 'program_start'+'keyboard_wait'+'pause'+'resume'+'program_end'
ann_comb2_added=delete_annotation(ann_comb2_added,['program_start','keyboard_wait','pause','resume','program_end'])
raw2.set_annotations(ann_comb2_added)

## delete 'TRIG-matlab', keep 'TRIG[001]:1'/'TRIG[001]:1:inserted' only
ann_natus_added=delete_annotation(ann_comb2_added,'TRIG-matlab')
raw2.set_annotations(ann_natus_added)

## epoch: 15 seconds after trigger
all_events, all_event_id = mne.events_from_annotations(raw2)
epochs = mne.Epochs(raw, all_events, tmin=0, tmax=15,baseline=None)

## sentence,delete 'Pausing' and 'Resuming'
prompts2=[]
for p in range(len(prompts)):
    if prompts[p][1] not in ['Pausing','Resuming']:
        if p<=len(prompts)-2 and prompts[p+1][1]!='Pausing': # A pause will repeat the previous unfinished sentence.
            prompts2.append(prompts[p][1])
        if p == len(prompts) - 1 and prompts[p][1]!='Pausing':
            prompts2.append(prompts[p][1])

len(epochs.events)==len(prompts2) # must be equal

## save everything: epoch and sentences
filename=data_dir+'processed/'+type+str(sid)+'/SEEG'+str(sid)+'-epo.fif'
epochs.save(filename, overwrite=True)
sentences=np.array(prompts2, dtype=object)
filename2=data_dir+'processed/'+type+str(sid)+'/sentences.npy'
np.save(filename2,sentences)
#sentences2=np.load(filename2,allow_pickle=True)
