'''
validate on particular one subject while test on another particular subject;
train on the rest subject;
'''

from sklearn.preprocessing import StandardScaler
from sEMG_Zhangbin.config import *

# LOO schema: test on one subject after training on the rest
def sub_split(taski=None, cat='ET',val_sub=None, test_sub=None): # test_on: 0:ET, 1:PD, 2:others, 3:NC; sub_id:use one perticular subject
    data_dir2=data_dir+'data2/'
    if taski is not None:#taski = [0]  # a list of task indexes
        strr = ""
        for i in taski:
            strr = strr + str(i) + "_"
        strr = strr[:-1]

        ET_data_raw = np.load(data_dir + 'ET_data_task_' + strr + '.npy', allow_pickle='TRUE').item()  # 5
        PD_data_raw = np.load(data_dir + 'PD_data_task_' + strr + '.npy', allow_pickle='TRUE').item()  # 6
        others_data_raw = np.load(data_dir + 'others_data_task_' + strr + '.npy', allow_pickle='TRUE').item()  # 4
        NC_data_raw = np.load(data_dir + 'NC_data_task_' + strr + '.npy', allow_pickle='TRUE').item()  # 18

    else:
        ET_data_raw = np.load(data_dir2+'ET_data.npy',allow_pickle='TRUE').item() # 5
        PD_data_raw = np.load(data_dir2+'PD_data.npy',allow_pickle='TRUE').item() # 6
        others_data_raw = np.load(data_dir2+'others_data.npy',allow_pickle='TRUE').item() # 4
        NC_data_raw = np.load(data_dir2+'NC_data.npy',allow_pickle='TRUE').item() # 18

    # TODO: if test_on=0: elif test_on=1 ....
    # test on ET subject
    #test_sub='TP007'
    print("Test on: "+test_sub+"; Validate on: "+val_sub+".")
    if cat=='ET':
        test_data_raw = ET_data_raw[test_sub]
        ET_data_raw.pop(test_sub)  # remove the subject in place
        val_data_raw = ET_data_raw[val_sub]
        ET_data_raw.pop(val_sub)  # remove the subject in place
    elif cat=='PD':
        test_data_raw = PD_data_raw[test_sub]
        PD_data_raw.pop(test_sub) # remove the subject in place
        val_data_raw = PD_data_raw[val_sub]
        PD_data_raw.pop(val_sub)  # remove the subject in place
    elif cat=='others':
        test_data_raw = others_data_raw[test_sub]
        others_data_raw.pop(test_sub)  # remove the subject in place
        val_data_raw = others_data_raw[val_sub]
        others_data_raw.pop(val_sub)  # remove the subject in place
    elif cat=='NC':
        test_data_raw = NC_data_raw[test_sub]
        NC_data_raw.pop(test_sub)  # remove the subject in place
        val_data_raw = NC_data_raw[val_sub]
        NC_data_raw.pop(val_sub)  # remove the subject in place

    ET_data_tmp=[]
    PD_data_tmp=[]
    others_data_tmp=[]
    NC_data_tmp=[]
    test_data_tmp=[]
    val_data_tmp=[]

    # [ 145 * (48000,7)]
    for sub in ET_data_raw.keys():
        for handi in ET_data_raw[sub]:
            #if 'L' in ET_data_raw[sub] or 'R' in ET_data_raw[sub]:
            for taski in ET_data_raw[sub][handi]:
                scaler = StandardScaler() # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                ET_data_tmp.append(taski)
    for sub in PD_data_raw.keys():
        for handi in PD_data_raw[sub]:
            #if 'L' in PD_data_raw[sub] or 'R' in PD_data_raw[sub]:
            for taski in PD_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                PD_data_tmp.append(taski)

    for sub in others_data_raw.keys():
        for handi in others_data_raw[sub]:
            #if 'L' in others_data_raw[sub] or 'R' in others_data_raw[sub]:
            for taski in others_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                others_data_tmp.append(taski)

    for sub in list(NC_data_raw.keys())[:4]:
        for handi in NC_data_raw[sub]:
            #if 'L' in NC_data_raw[sub] or 'R' in NC_data_raw[sub]:
            for taski in NC_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                NC_data_tmp.append(taski)

    for handi in test_data_raw:
        #if 'L' in test_data_raw or 'R' in test_data_raw:
        for taski in test_data_raw[handi]:
            scaler = StandardScaler()  # input shape: (n_sample, n_feature)
            taski = scaler.fit_transform((taski))
            test_data_tmp.append(taski)

    for handi in val_data_raw:
        #if 'L' in val_data_raw or 'R' in val_data_raw:
        for taski in val_data_raw[handi]:
            scaler = StandardScaler()  # input shape: (n_sample, n_feature)
            taski = scaler.fit_transform((taski))
            val_data_tmp.append(taski)

    return ET_data_tmp,PD_data_tmp,others_data_tmp,NC_data_tmp,test_data_tmp, val_data_tmp

# LOO schema: test on one subject after training on the rest
def sub_split2(taski=None, cat='ET',val_sub=None, test_sub=None): # test_on: 0:ET, 1:PD, 2:others, 3:NC; sub_id:use one perticular subject
    data_dir2=data_dir+'data2/'
    if taski is not None:#taski = [0]  # a list of task indexes
        strr = ""
        for i in taski:
            strr = strr + str(i) + "_"
        strr = strr[:-1]

        ET_data_raw = np.load(data_dir + 'ET_data_task_' + strr + '.npy', allow_pickle='TRUE').item()  # 5
        PD_data_raw = np.load(data_dir + 'PD_data_task_' + strr + '.npy', allow_pickle='TRUE').item()  # 6
        others_data_raw = np.load(data_dir + 'others_data_task_' + strr + '.npy', allow_pickle='TRUE').item()  # 4
        NC_data_raw = np.load(data_dir + 'NC_data_task_' + strr + '.npy', allow_pickle='TRUE').item()  # 18

    else:
        ET_data_raw = np.load(data_dir2+'ET_data.npy',allow_pickle='TRUE').item() # 5
        PD_data_raw = np.load(data_dir2+'PD_data.npy',allow_pickle='TRUE').item() # 6
        others_data_raw = np.load(data_dir2+'others_data.npy',allow_pickle='TRUE').item() # 4
        NC_data_raw = np.load(data_dir2+'NC_data.npy',allow_pickle='TRUE').item() # 18

    # TODO: if test_on=0: elif test_on=1 ....
    # test on ET subject
    #test_sub='TP007'
    print("Test on: "+test_sub+"; Validate on: "+val_sub+".")
    if cat=='ET':
        test_data_raw = ET_data_raw[test_sub]
        test_data_labels=ET_task_labels[test_sub]
        ET_data_raw.pop(test_sub)  # remove the subject in place
        val_data_raw = ET_data_raw[val_sub]
        val_data_labels = ET_task_labels[val_sub]
        ET_data_raw.pop(val_sub)  # remove the subject in place
    elif cat=='PD':
        test_data_raw = PD_data_raw[test_sub]
        test_data_labels = PD_task_labels[test_sub]
        PD_data_raw.pop(test_sub) # remove the subject in place
        val_data_raw = PD_data_raw[val_sub]
        val_data_labels = PD_task_labels[val_sub]
        PD_data_raw.pop(val_sub)  # remove the subject in place
    elif cat=='others':
        test_data_raw = others_data_raw[test_sub]
        test_data_labels = others_task_labels[test_sub]
        others_data_raw.pop(test_sub)  # remove the subject in place
        val_data_raw = others_data_raw[val_sub]
        val_data_labels = others_task_labels[val_sub]
        others_data_raw.pop(val_sub)  # remove the subject in place
    elif cat=='NC':
        test_data_raw = NC_data_raw[test_sub]
        test_data_labels = NC_task_labels[test_sub]
        NC_data_raw.pop(test_sub)  # remove the subject in place
        val_data_raw = NC_data_raw[val_sub]
        val_data_labels = NC_task_labels[val_sub]
        NC_data_raw.pop(val_sub)  # remove the subject in place

    ET_data_tmp=[]
    PD_data_tmp=[]
    others_data_tmp=[]
    NC_data_tmp=[]
    test_data_tmp=[]
    val_data_tmp=[]

    # [ 145 * (48000,7)]
    for sub in ET_data_raw.keys():
        for handi in ET_data_raw[sub]:
            assert len(ET_data_raw[sub][handi])==len(ET_task_labels[sub][handi])
            #if 'L' in ET_data_raw[sub] or 'R' in ET_data_raw[sub]:
            for i, taski in enumerate(ET_data_raw[sub][handi]):
                scaler = StandardScaler() # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                ET_data_tmp.append(taski)
    for sub in PD_data_raw.keys():
        for handi in PD_data_raw[sub]:
            assert len(PD_data_raw[sub][handi]) == len(PD_task_labels[sub][handi])
            #if 'L' in PD_data_raw[sub] or 'R' in PD_data_raw[sub]:
            for taski in PD_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                PD_data_tmp.append(taski)

    for sub in others_data_raw.keys():
        for handi in others_data_raw[sub]:
            assert len(others_data_raw[sub][handi]) == len(others_task_labels[sub][handi])
            #if 'L' in others_data_raw[sub] or 'R' in others_data_raw[sub]:
            for taski in others_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                others_data_tmp.append(taski)

    for sub in list(NC_data_raw.keys())[:4]:
        for handi in NC_data_raw[sub]:
            assert len(NC_data_raw[sub][handi]) == len(NC_task_labels[sub][handi])
            #if 'L' in NC_data_raw[sub] or 'R' in NC_data_raw[sub]:
            for taski in NC_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                NC_data_tmp.append(taski)

    test_data_labels2 = []
    for handi in test_data_raw:
        assert len(test_data_raw[handi]) == len(test_data_labels[handi])
        #if 'L' in test_data_raw or 'R' in test_data_raw:
        for i,taski in enumerate(test_data_raw[handi]):
            test_data_labels2.append(handi+str(test_data_labels[handi][i]))
            scaler = StandardScaler()  # input shape: (n_sample, n_feature)
            taski = scaler.fit_transform((taski))
            test_data_tmp.append(taski)

    val_data_labels2 = []
    for handi in val_data_raw:
        assert len(val_data_raw[handi]) == len(val_data_labels[handi])
        #if 'L' in val_data_raw or 'R' in val_data_raw:
        for i,taski in enumerate(val_data_raw[handi]):
            val_data_labels2.append(handi + str(val_data_labels[handi][i]))
            scaler = StandardScaler()  # input shape: (n_sample, n_feature)
            taski = scaler.fit_transform((taski))
            val_data_tmp.append(taski)

    return ET_data_tmp,PD_data_tmp,others_data_tmp,NC_data_tmp,test_data_tmp,test_data_labels2, val_data_tmp,val_data_labels2


# concatenate all subject
def windowing(data, wind_size,stride): #data shape: (48000, 7)
    windowed=[]
    s = 0
    total_len=data.shape[0]
    while stride * s + wind_size < total_len:
        start = s * stride
        tmp = data[start:(start + wind_size),:]
        windowed.append(tmp)
        s = s + 1
    # add the last window
    last_s = s - 1
    if stride * last_s + wind_size < total_len - 100: # discard the rest if too short data remaining
        tmp = data[-wind_size:,:]
        windowed.append(tmp)

    return np.asarray(windowed)

def sub_split_one_task(test_on=0,sub_id=0): # test_on: 0:ET, 1:PD, 2:others, 3:NC; sub_id:use one perticular subject
    ET_data_raw = np.load(data_dir+'ET_data.npy',allow_pickle='TRUE').item() # 5
    PD_data_raw = np.load(data_dir+'PD_data.npy',allow_pickle='TRUE').item() # 6
    others_data_raw = np.load(data_dir+'others_data.npy',allow_pickle='TRUE').item() # 4
    NC_data_raw = np.load(data_dir+'NC_data.npy',allow_pickle='TRUE').item() # 18

    # TODO: if test_on=0: elif test_on=1 ....
    # test on ET subject
    sid=random.sample(list(range(len(PD_data_raw))),1)
    sub=list(PD_data_raw.keys())[sid[0]]
    test_data_raw = PD_data_raw[sub]
    PD_data_raw.pop(sub) # remove the subject in place
    # validate on PD subject
    sid = random.sample(list(range(len(PD_data_raw))), 1)
    sub = list(PD_data_raw.keys())[sid[0]]
    val_data_raw = PD_data_raw[sub]
    PD_data_raw.pop(sub)  # remove the subject in place


    ET_data_tmp=[]
    PD_data_tmp=[]
    others_data_tmp=[]
    NC_data_tmp=[]
    test_data_tmp=[]
    val_data_tmp=[]

    # [ 145 * (48000,7)]
    for sub in ET_data_raw.keys():
        for handi in ET_data_raw[sub]:
            #if 'L' in ET_data_raw[sub] or 'R' in ET_data_raw[sub]:
            for taski in ET_data_raw[sub][handi]:
                scaler = StandardScaler() # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                ET_data_tmp.append(taski)
    for sub in PD_data_raw.keys():
        for handi in PD_data_raw[sub]:
            #if 'L' in PD_data_raw[sub] or 'R' in PD_data_raw[sub]:
            for taski in PD_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                PD_data_tmp.append(taski)

    for sub in others_data_raw.keys():
        for handi in others_data_raw[sub]:
            #if 'L' in others_data_raw[sub] or 'R' in others_data_raw[sub]:
            for taski in others_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                others_data_tmp.append(taski)

    for sub in NC_data_raw.keys():
        for handi in NC_data_raw[sub]:
            #if 'L' in NC_data_raw[sub] or 'R' in NC_data_raw[sub]:
            for taski in NC_data_raw[sub][handi]:
                scaler = StandardScaler()  # input shape: (n_sample, n_feature)
                taski = scaler.fit_transform((taski))
                NC_data_tmp.append(taski)

    for handi in test_data_raw:
        #if 'L' in test_data_raw or 'R' in test_data_raw:
        for taski in test_data_raw[handi]:
            scaler = StandardScaler()  # input shape: (n_sample, n_feature)
            taski = scaler.fit_transform((taski))
            test_data_tmp.append(taski)

    for handi in val_data_raw:
        #if 'L' in val_data_raw or 'R' in val_data_raw:
        for taski in val_data_raw[handi]:
            scaler = StandardScaler()  # input shape: (n_sample, n_feature)
            taski = scaler.fit_transform((taski))
            val_data_tmp.append(taski)

    return ET_data_tmp,PD_data_tmp,others_data_tmp,NC_data_tmp,test_data_tmp, val_data_tmp

