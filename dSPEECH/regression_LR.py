import copy, resampy
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dSPEECH.config import *
from dSPEECH.config import opt_regression_LR as opt
from scipy.io import wavfile
from speech_Dutch.baseline_linear_regression.extract_features import extractHG, stackFeatures, extractMelSpecs
import matplotlib.pyplot as plt

modality='SEEG'
task='speak' # 'listen'/'speak'/'imagine'
sid=2
eeg_sr=1024
result_dir = data_dir + 'result/regression/'+modality+str(sid)+'/linear_regression/'+task+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
## load epochs and sentences
filename=data_dir+'processed/'+modality+str(sid)+'/'+modality+str(sid)+'-epo.fif'
epochs=mne.read_epochs(filename)
filename2=data_dir+'processed/'+modality+str(sid)+'/sentences.npy'
sentences=np.load(filename2,allow_pickle=True)

data=epochs.get_data() # (100, 149, 15361)
if task=='listen':
    eeg = data[:, :, 0:5 * eeg_sr]
elif task=='speak':
    eeg = data[:, :, 5 * eeg_sr:10 * eeg_sr]  # speaking part only (100, 149, 5120)
elif task=='imagine':
    eeg = data[:, :, 10 * eeg_sr:15 * eeg_sr]

winlen=opt['winlen']
frameshift=opt['frameshift']
melbins=opt['melbins']
stepsize=opt['stepsize']
modelorder=opt['modelorder']
print('Stacking features using order:' + str(modelorder) + ', and step size:' + str(stepsize) + '.')

feats=[]
melspecs=[]
for i in range(eeg.shape[0]):
    eegi=eeg[i,:,:]
    audio_file=data_dir+'/paradigm_audio/5_seconds_original/'+str(i+1)+'.wav'
    audio_sr, audio = wavfile.read(audio_file) # (240000,)
    feat = extractHG(eegi.transpose(), eeg_sr, windowLength=winlen, frameshift=frameshift) # (495, 149)
    feat = stackFeatures(feat, modelOrder=modelorder, stepSize=stepsize)  # (487, 1341)

    # Process Audio
    scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)  # 32767??
    melSpec = extractMelSpecs(scaled, audio_sr, melbins=melbins, windowLength=winlen, frameshift=frameshift) # (495, 23)
    # Align to EEG features
    melSpec = melSpec[modelorder * stepsize:melSpec.shape[0] - modelorder * stepsize, :] # (487, 23)

    # adjust length (differences might occur due to rounding in the number of windows)
    if melSpec.shape[0] != feat.shape[0]:
        print('EEG and mel difference:' + str(melSpec.shape[0] - feat.shape[0]) + '.')
        tLen = np.min([melSpec.shape[0], feat.shape[0]])
        melSpec = melSpec[:tLen, :]  # (25863, 80)
        feat = feat[:tLen, :]  # (25863, 127)
    else:
        pass
        #print('EEG and mel lengths are the same. ')

    feats.append(feat)
    melspecs.append(melSpec)

# flatten list of 2d to 3d
tmp=np.asarray(feats) # (100 trial, 487 time, 1341 channel/feature)
tmp=tmp.transpose(2,0,1) # -->(1341 channel, 100 trial, 487 time)
feats=np.reshape(tmp,(tmp.shape[0],-1)) # (1341, 48700)

tmp=np.asarray(melspecs) # (100 trial, 487 time, 23 channel)
tmp=tmp.transpose(2,0,1) # (23 channl, 100 trial, 487 time)
melspecs=np.reshape(tmp,(tmp.shape[0],-1)) # (23, 48700)


nfolds = 5
kf = KFold(nfolds,shuffle=False)
est = LinearRegression(n_jobs=5)
pca = PCA()
numComps = 80

melspecs=melspecs.transpose()#  (n_samples, n_features)
feats=feats.transpose()
#Initialize an empty spectrogram to save the reconstruction to
rec_spec = np.zeros(melspecs.shape)
#Save the correlation coefficients for each fold
rs = np.zeros((nfolds,melspecs.shape[1]))
explainedVariance = np.zeros((1,nfolds))

for k,(train, test) in enumerate(kf.split(feats)):
    # train,test=next(iter(kf.split(feats)))
    #Z-Normalize with mean and std from the training data
    mu=np.mean(feats[train,:],axis=0)
    std=np.std(feats[train,:],axis=0)
    trainData=(feats[train,:]-mu)/std
    testData=(feats[test,:]-mu)/std

    #Fit PCA to training data
    pca.fit(trainData)
    #Get percentage of explained variance by selected components
    explainedVariance[0,k] =  np.sum(pca.explained_variance_ratio_[:numComps])
    #Tranform data into component space
    trainData=np.dot(trainData, pca.components_[:numComps,:].T)
    testData = np.dot(testData, pca.components_[:numComps,:].T)

    #Fit the regression model
    est.fit(trainData, melspecs[train, :])
    #Predict the reconstructed spectrogram for the test data
    rec_spec[test, :] = est.predict(testData)

    #Evaluate reconstruction of this fold
    for specBin in range(melspecs.shape[1]):
        if np.any(np.isnan(rec_spec)):
            print('%s has %d broken samples in reconstruction' % (modality+str(sid), np.sum(np.isnan(rec_spec))))
        r, p = pearsonr(melspecs[test, specBin], rec_spec[test, specBin])
        rs[k,specBin] = r
    #break
print('%s: mean correlation of %f' % (modality+str(sid),np.mean(rs)))

estimate_random_baseline=True
numRands=500
randomControl = np.zeros((numRands, melbins))
if estimate_random_baseline:
    print("Estimating random baseline.")
    #Estimate random baseline
    for randRound in range(numRands):
        #Choose a random splitting point at least 10% of the dataset size away
        splitPoint = np.random.choice(np.arange(int(melspecs.shape[0]*0.1),int(melspecs.shape[0]*0.9)))
        #Swap the dataset on the splitting point
        shuffled = np.concatenate((melspecs[splitPoint:,:],melspecs[:splitPoint,:]))
        #Calculate the correlations
        for specBin in range(melspecs.shape[1]):
            if np.any(np.isnan(rec_spec)):
                print('%s has %d broken samples in reconstruction' % (modality+str(sid), np.sum(np.isnan(rec_spec))))
            r, p = pearsonr(melspecs[:,specBin], shuffled[:,specBin])
            randomControl[randRound,specBin]=r
    random_corr=np.mean(randomControl[:,:])
    print('Random control has mean correlation of %f' % (random_corr))

pred,truth=rec_spec[test,:],melspecs[test,:]
fig,ax=plt.subplots(2,1)
np.save(result_dir + 'pred.npy', pred)
np.save(result_dir + 'truth.npy', truth)
ax[0].imshow(truth[:, :].transpose(), aspect='auto')
ax[1].imshow(pred[:, :].transpose(), aspect='auto')
fig.savefig(result_dir+'result.png')
mse=mean_squared_error(truth[:3000, :], pred[:3000, :])
corr=np.mean(rs)
#exit()  # stop from here
filename = result_dir + 'result.txt'
with open(filename, 'w') as f:
    f.write('corr:')
    f.write(str(corr))
    f.write('\n')
    f.write('mse:')
    f.write(str(mse))
    if estimate_random_baseline:
        f.write('\n')
        f.write('Random corr:')
        f.write(str(random_corr))
ax[0].clear()
ax[1].clear()



