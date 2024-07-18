'''
Paradigm Version 2;
compare triggers from matlab logfile and Natus;
Hospital laptop is about 10 seconds behind (Needs to be verified);
Subjects: ECoG1;
'''
import calendar
import datetime
import glob
from utils.util_MNE import delete_annotation, keep_annotation
from dSPEECH.config import *

sf=1024
type='ECoG' #'SEEG/ECoG
sid=1
eeg_file=data_dir+'raw/'+type+str(sid)+'_*/EEG.edf' # SEEG1_datetime
eeg_file=os.path.normpath(glob.glob(eeg_file)[0])
raw = mne.io.read_raw_edf(eeg_file)
plot_scale=100e-5
plot_channels=5
raw.plot(duration=200,time_format='clock',n_channels=plot_channels,scalings=dict(eeg=plot_scale))
raw.info.get_channel_types() # All chs are type of 'eeg'
##  extract annotations='TRIG[001]:1'
anno_natus=raw.annotations
if type=='ECoG':
    event_descripion='Patient Event'
else:
    event_descripion = 'TRIG[001]:1'


anno_natus=keep_annotation(anno_natus,event_descripion)
raw.set_annotations(anno_natus)
#envs=mne.events_from_annotations(raw)

## 1, export prompts to inf.txt from matlab using export_logfile_to_text.m file;
# 2, Manually add time stamp to 'Resumeing' event. The time of resuming is the same as that of next sentence;
# 3, Read inf.txt, and format it to lines in the form of "time, sentence";
filename=data_dir+'raw/'+type+str(sid)+'_*/matlab/inf.txt'
filename=os.path.normpath(glob.glob(filename)[0])
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
cal_dict={month: index for index, month in enumerate(calendar.month_abbr) if month}
orig_time=anno_natus[0]['orig_time']#datetime.datetime(2023, 11, 17, 9, 51, 34, tzinfo=datetime.timezone.utc)
onsets=[]
durations=[]
descriptions=[]
for prompt in prompts:
    time=prompt[0]
    year=int(time.split(' ')[0].split('-')[-1])
    month_tmp = time.split(' ')[0].split('-')[1]
    month=cal_dict[month_tmp]
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
ann_extra=anno_natus.__add__(anno_matlab)
raw1=raw.copy()
raw1.set_annotations(ann_extra)

## visually inspect the correspondence between Natus triggers and matlab triggers;
# Add two annotations: program_start, program_end in annotation mode;
# (1,Press 'a' enter annotation mode; 2, type program_start, then hit enter; 3, click and drag will create program_start point;
# 4, type program_end, then hit enter; 5, click and drag will create program_end point;)
# The raw1.annotation will reflect the modified annotation in real-time (no need to close the figure);
raw1.plot(duration=100,time_format='clock',n_channels=plot_channels,scalings=dict(eeg=plot_scale))


## extract the experimental data by only keeping program_start and program_end triggers;
seg_anno=raw1.annotations.copy()
seg_anno=keep_annotation(seg_anno, ['program_start','program_end'])
# get start time from 'program_start' trigger
tmin=seg_anno[0]['onset']-5 # extra 5 seconds
tmax=seg_anno[1]['onset']+seg_anno[1]['duration']+5 # extra 5 seconds
raw2=raw1.crop(tmin=tmin,tmax=tmax)
#raw2=raw1.crop_by_annotations(seg_anno) # this will only return data within each annotation period
raw2.plot(duration=100,time_format='float',n_channels=plot_channels,scalings=dict(eeg=plot_scale)) # 'float'/'clock'

## SEEG1: replace the first 'TRIG[001]:1' with 'keyboard_wait'
anno_tmp=raw2.annotations.copy()
replace_anno=mne.Annotations(onset=anno_tmp[1]['onset'],duration=anno_tmp[1]['duration'],description='keyboard_wait',orig_time=anno_tmp[1]['orig_time'])
anno_tmp.delete(1) # the second trigger(first 'TRIG[001]:1' trigger)
anno_natus2=anno_tmp+replace_anno
raw2.set_annotations(anno_natus2)

## deal with the missing natus triggers
n_natus=[tmp for tmp in raw2.annotations if tmp['description']=='TRIG[001]:1' ]
n_matlab=[tmp for tmp in raw2.annotations if tmp['description']=='TRIG-matlab' ]
diff=len(n_matlab)-len(n_natus) # 4 triggers missing in n_natus
# visual check the missing natus triggers
raw2.plot(duration=100,time_format='float',n_channels=plot_channels,scalings=dict(eeg=plot_scale)) # 'float'/'clock'
# frame the missing triggers
missings=[[321.051,337.348], [753.128,769.356], [784.389,800.344], [1229.153,1245.415]]
raw2.annotations[20]['onset']-raw2.annotations[18]['onset'] # 15.25/15.125/15.11/15.13/15.125/15.127/15.132/15.135/15.132/
# calculate the missing onset
insert=[]
for m in missings:
    for n in n_natus:
        if m[0]+tmin<n['onset']+15<m[1]+tmin:
            insert.append(n['onset']+15.125) #s
# create missing annotations
onsets=[]
durations=[]
descriptions=[]
for i in insert:
    onsets.append(i)
    durations.append(0)
    descriptions.append('TRIG[001]:1:inserted')
anno_missing=mne.Annotations(onset=onsets,duration=durations,description=descriptions,orig_time=orig_time)# 106=102triggers+4pauses
ann_natus_added=anno_natus2.__add__(anno_missing)
raw2.set_annotations(ann_natus_added)
#check result: contains annotations of 'program_start'+'keyboard_wait'+'TRIG[001]:1'+'TRIG[001]:1:inserted'+'pause'+'resume'+'program_end'
raw2.plot(duration=100,time_format='float',n_channels=plot_channels,scalings=dict(eeg=plot_scale)) # 'float'/'clock'

## delete 'program_start'+'keyboard_wait'+'pause'+'resume'+'program_end'
ann_natus_added=delete_annotation(ann_natus_added,['program_start','keyboard_wait','pause','resume','program_end'])
raw2.set_annotations(ann_natus_added)

## delete 'TRIG-matlab', keep 'TRIG[001]:1'/'TRIG[001]:1:inserted' only
ann_natus_added=delete_annotation(ann_natus_added,'TRIG-matlab')
raw2.set_annotations(ann_natus_added)

## epoch: 15 seconds after trigger
all_events, all_event_id = mne.events_from_annotations(raw2)
epochs = mne.Epochs(raw, all_events, tmin=0, tmax=15,baseline=None)

## sentence
prompts2=[]
for p in prompts:
    if p[1] not in ['Pausing','Resuming']:
        prompts2.append(p[1])

len(epochs.events)==len(prompts2) # must be equal

## save everything: epoch and sentences
filename=data_dir+'processed/'+type+str(sid)+'/SEEG1-epo.fif'
epochs.save(filename, overwrite=True)
sentences=np.array(prompts2, dtype=object)
filename2=data_dir+'processed/'+type+str(sid)+'/sentences.npy'
np.save(filename2,sentences)
#sentences2=np.load(filename2,allow_pickle=True)


fig,ax=plt.subplots()
