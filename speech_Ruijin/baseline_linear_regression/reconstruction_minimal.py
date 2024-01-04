'''
This script use a simple linear regression as a baseline;
'''

import os

import scipy
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from pre_all import computer
import example_speech.SingleWordProductionDutch.reconstructWave as rW
import example_speech.SingleWordProductionDutch.MelFilterBank as mel
from speech_Ruijin.baseline_linear_regression.extract_features import dataset, extractHG, stackFeatures, extractMelSpecs
from speech_Ruijin.config import data_dir, extra_EEG_extracting


def createAudio(spectrogram, audiosr=16000, winLength=0.05, frameshift=0.01,log=True):
    """
    Create a reconstructed audio wavefrom
    
    Parameters
    ----------
    spectrogram: array (time, bins)
        Spectrogram of the audio
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram was calculated
    frameshift: float
        Shift (in seconds) after which next window was extracted
    Returns
    ----------
    scaled: array
        Scaled audio waveform
    """
    nfolds = 10
    hop = int(spectrogram.shape[0]/nfolds)
    rec_audio = np.array([])
    if log:
        mfb = mel.MelFilterBank(int((audiosr * winLength) / 2 + 1), spectrogram.shape[1], audiosr)
        for_reconstruction = mfb.fromLogMels(spectrogram)
    else:
        for_reconstruction = spectrogram
    for w in range(0,spectrogram.shape[0],hop):
        spec = for_reconstruction[w:min(w+hop,for_reconstruction.shape[0]),:]
        rec = rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
        rec_audio = np.append(rec_audio,rec)
    scaled = np.int16(rec_audio/np.max(np.abs(rec_audio)) * 32767)
    #scaled = rec_audio / np.max(np.abs(rec_audio))
    return scaled



def baseline(data_name=None,sid=1,session=1,test_shift=False,test_shifts=None,model_order=None, step_size=None,
             melbins=23,winL=0.05, frameshift=0.01,estimate_random_baseline=False,save_result=False):
    if computer=='mac':
        feat_path = r'/Volumes/Samsung_T5/data/SingleWordProductionDutch/features'
        result_path = r'/Volumes/Samsung_T5/data/SingleWordProductionDutch/results'
    else:
        feat_path = r'H:\Long\data\SingleWordProductionDutch-iBIDS\features'
        result_path = r'H:\Long\data\SingleWordProductionDutch-iBIDS\results'
    audiosr = 16000

    nfolds = 5
    kf = KFold(nfolds,shuffle=False)
    est = LinearRegression(n_jobs=5)
    pca = PCA()
    numComps = 50
    
    #Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    allRes = np.zeros((1,nfolds,melbins))
    explainedVariance = np.zeros((1,nfolds))
    numRands = 100
    randomControl = np.zeros((1,numRands, melbins))

    if test_shift:
        #model_order=4
        #winL = 0.05
        #frameshift = 0.01
        #stepSize = 5

        from scipy.io import wavfile
        from speech_Ruijin.config import data_dir
        filename = data_dir + "P" + str(sid) + "/processed/EEG/session" + str(session) + "_curry_crop.npy"
        print('Loading ' + filename + '.')
        eeg_tmp = np.load(filename)
        #eeg_tmp=eeg_tmp[:,5000:]
        eeg_sr = 1000
        corrs = []
        filename = data_dir + "P" + str(sid) + "/processed/audio/session" + str(session) + "_denoised.wav"
        print('Load ' + filename + '.')
        audio_sr, audio = wavfile.read(filename)
        target_SR = 16000
        audio = scipy.signal.decimate(audio, int(audio_sr / target_SR))  # (4805019,)
        audio_sr = target_SR
        scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)  # 32767??
        #scaled=scaled[5*16000:]
        spectrogram = extractMelSpecs(scaled, audio_sr, melbins=melbins, windowLength=winL, frameshift=frameshift)  #
        spectrogram = spectrogram[model_order * step_size:spectrogram.shape[0] - model_order * step_size, :]  # (29986, 23)

        for i, test_shift in enumerate(test_shifts):
            assert test_shift < extra_EEG_extracting * eeg_sr
            if test_shift==0:
                eeg = eeg_tmp.transpose()
            else:
                eeg = eeg_tmp[:,extra_EEG_extracting * eeg_sr + test_shift:-extra_EEG_extracting * eeg_sr + test_shift].transpose()  # extra extra_EEG_extracting second from beginning and ending

            # Extract HG features and average the window
            data = extractHG(eeg, eeg_sr, windowLength=winL, frameshift=frameshift)  # (307523, 127)-->(30026, 127)
            data = stackFeatures(data, modelOrder=model_order, stepSize=step_size)  # (29986, 1143)

            # adjust length (differences might occur due to rounding in the number of windows)
            if spectrogram.shape[0] != data.shape[0]:
                difference=spectrogram.shape[0]-data.shape[0]
                print('Differ: '+str(difference)+'.')
                tLen = np.min([spectrogram.shape[0], data.shape[0]])
                spectrogram = spectrogram[:tLen, :]
                data = data[:tLen, :]

            # Initialize an empty spectrogram to save the reconstruction to
            rec_spec = np.zeros(spectrogram.shape)
            # Save the correlation coefficients for each fold
            rs = np.zeros((nfolds, spectrogram.shape[1]))
            for k, (train, test) in enumerate(kf.split(data)):
                # Z-Normalize with mean and std from the training data
                mu = np.mean(data[train, :], axis=0)
                std = np.std(data[train, :], axis=0)
                trainData = (data[train, :] - mu) / std
                testData = (data[test, :] - mu) / std

                # Fit PCA to training data
                pca.fit(trainData)
                # Get percentage of explained variance by selected components
                explainedVariance[0, k] = np.sum(pca.explained_variance_ratio_[:numComps])
                # Tranform data into component space
                trainData = np.dot(trainData, pca.components_[:numComps, :].T)
                testData = np.dot(testData, pca.components_[:numComps, :].T)

                # Fit the regression model
                est.fit(trainData, spectrogram[train, :])
                # Predict the reconstructed spectrogram for the test data
                rec_spec[test, :] = est.predict(testData)

                # Evaluate reconstruction of this fold
                for specBin in range(spectrogram.shape[1]):
                    if np.any(np.isnan(rec_spec)):
                        print('%s has %d broken samples in reconstruction' % (sid, np.sum(np.isnan(rec_spec))))
                    r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                    rs[k, specBin] = r
                # break
                # Show evaluation result
            print('Shift: ' + str(test_shift) + '.')
            print('%s has mean correlation of %f' % (sid, np.mean(rs)))

            if i==0:
                best_corr = np.mean(rs)
                best_truth=spectrogram[test, :]
                best_pred=rec_spec[test, :]
                best_shift=test_shift
            else:
                if np.mean(rs)>best_corr:
                    best_corr = np.mean(rs)
                    best_truth = spectrogram[test, :]
                    best_pred = rec_spec[test, :]
                    best_shift = test_shift
        print('Best shift: '+str(best_shift)+'.')
        ax[0].imshow(best_truth.transpose(), cmap='RdBu_r', aspect='auto')
        ax[1].imshow(best_pred.transpose(), cmap='RdBu_r', aspect='auto')
        folder = data_dir + 'baseline_LR/mydata/mel_' + str(melbins) + '/sid' + str(sid) + '/session'+str(session)+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(folder + 'result.png')
        mse = mean_squared_error(best_truth, best_pred)
        filename = folder + 'result.txt'
        with open(filename, 'w') as f:
            f.write('corr:')
            f.write(str(best_corr))
            f.write('\n')
            f.write('mse:')
            f.write(str(mse))
            f.write('\n')
            f.write('best shift:')
            f.write(str(best_shift))
        ax[0].clear()
        ax[1].clear()

        return best_pred,best_truth,best_corr

    else:
        #Load the data
        pt = 'sub-' + f"{sid:02d}"
        print("Running for "+pt+'.')
        data,spectrogram=dataset(dataset_name=data_name,sid=sid,session=session,melbins=melbins,modelOrder=model_order,
                                 stepSize=step_size,winL=winL, frameshift=frameshift)

        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            #Z-Normalize with mean and std from the training data
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            trainData=(data[train,:]-mu)/std
            testData=(data[test,:]-mu)/std

            #Fit PCA to training data
            pca.fit(trainData)
            #Get percentage of explained variance by selected components
            explainedVariance[0,k] =  np.sum(pca.explained_variance_ratio_[:numComps])
            #Tranform data into component space
            trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)

            #Fit the regression model
            est.fit(trainData, spectrogram[train, :])
            #Predict the reconstructed spectrogram for the test data
            rec_spec[test, :] = est.predict(testData)

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r
            #break
        #Show evaluation result
        print('Shift: '+str(test_shift)+'.')
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[0,:,:]=rs

        if estimate_random_baseline:
            #Estimate random baseline
            for randRound in range(numRands):
                #Choose a random splitting point at least 10% of the dataset size away
                splitPoint = np.random.choice(np.arange(int(spectrogram.shape[0]*0.1),int(spectrogram.shape[0]*0.9)))
                #Swap the dataset on the splitting point
                shuffled = np.concatenate((spectrogram[splitPoint:,:],spectrogram[:splitPoint,:]))
                #Calculate the correlations
                for specBin in range(spectrogram.shape[1]):
                    if np.any(np.isnan(rec_spec)):
                        print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                    r, p = pearsonr(spectrogram[:,specBin], shuffled[:,specBin])
                    randomControl[1, randRound,specBin]=r
            print('Random control has mean correlation of %f' % (np.mean(randomControl[1,:,:])))

        if save_result:
            #Save reconstructed spectrogram
            os.makedirs(os.path.join(result_path), exist_ok=True)
            np.save(os.path.join(result_path, f'{pt}_orig_spec.npy'), spectrogram)
            np.save(os.path.join(result_path,f'{pt}_predicted_spec.npy'), rec_spec)

            #Synthesize waveform from spectrogram using Griffin-Lim
            reconstructedWav = createAudio(rec_spec,audiosr=audiosr,winLength=winL,frameshift=frameshift)
            wavfile.write(os.path.join(result_path,f'{pt}_predicted.wav'),int(audiosr),reconstructedWav)

            #For comparison synthesize the original spectrogram with Griffin-Lim
            origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winL,frameshift=frameshift)
            wavfile.write(os.path.join(result_path,f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)
        #break # just run for 1 fold
        return rec_spec[test,:],spectrogram[test,:],rs

    #Save results in numpy arrays          
    #np.save(os.path.join(result_path,'linearResults.npy'),allRes)
    #np.save(os.path.join(result_path,'randomResults.npy'),randomControl)
    #np.save(os.path.join(result_path,'explainedVariance.npy'),explainedVariance)

if __name__=="__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    from speech_Ruijin.baseline_linear_regression.opt import opt

    model_order = opt['model_order']  # 4 # 4
    step_size = opt['step_size']  # 4 # 5:0.82 1:0.76
    winL = opt['winL']
    frameshift = opt['frameshift']
    melbins=23
    data_name='SingleWordProductionDutch' # 'SingleWordProductionDutch'/'mydata'/'Huashan'
    if data_name=='mydata':
        sid=5
        session=1
        sf_EEG = 1000
        test_alignment = True
        if test_alignment:
            test_shift_start = int(-0.1 * sf_EEG)  # -0.999
            test_shift_end = int(0.1 * sf_EEG)  # 0.999
            test_shifts = np.arange(test_shift_start, test_shift_end, 50)
            test_shifts=[0,]
            pred, truth, rs = baseline(data_name=data_name, sid=sid, session=session, test_shift=True,test_shifts=test_shifts
                                       ,model_order=model_order,step_size=step_size,estimate_random_baseline=False)
            exit()
    elif data_name=='SingleWordProductionDutch' or data_name=='Huashan':
        sids=[1,2,3,4,5,6,7,8,9,10]
        #sids = [3,]
        for sid in sids:
            folder = data_dir + 'baseline_LR/SingleWordProductionDutch/mel_'+str(melbins)+'/sid' + str(sid) + '/'
            print('Result folder: '+folder+'.')
            if not os.path.exists(folder):
                os.makedirs(folder)

            pred, truth, rs = baseline(data_name=data_name,sid=sid,melbins=melbins,model_order=model_order,
                                       step_size=step_size, estimate_random_baseline=False,winL=winL, frameshift=frameshift)
            np.save(folder + 'pred.npy', pred)
            np.save(folder + 'truth.npy', truth)
            ax[0].imshow(truth[:, :].transpose(), aspect='auto')
            ax[1].imshow(pred[:, :].transpose(), aspect='auto')
            fig.savefig(folder+'result.png')
            mse=mean_squared_error(truth[:3000, :], pred[:3000, :])
            corr=np.mean(rs)
            #exit()  # stop from here
            filename = folder + 'result.txt'
            with open(filename, 'w') as f:
                f.write('corr:')
                f.write(str(corr))
                f.write('\n')
                f.write('mse:')
                f.write(str(mse))
            ax[0].clear()
            ax[1].clear()
    '''
    # test the 3D plot
    X = np.arange(5997)
    Y=np.arange(23)
    X, Y = np.meshgrid(X, Y)
    from matplotlib import cm
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax1.plot_surface(X, Y, pred.transpose(), cmap=cm.coolwarm,linewidth=0, antialiased=True)
    surf1 = ax1.plot_surface(X, Y, truth.transpose(), linewidth=0, antialiased=True)
    '''
    exit() # stop from here

    # or just load the saved result
    truth=np.load('/Volumes/Samsung_T5/data/SingleWordProductionDutch/results/sub-03_orig_spec.npy')
    pred=np.load('/Volumes/Samsung_T5/data/SingleWordProductionDutch/results/sub-03_predicted_spec.npy')

    audiosr=16000
    winLength = 0.05
    frameshift = 0.01
    pred2= createAudio(pred[:3000, :], audiosr=audiosr, winLength=winLength, frameshift=frameshift)
    truth2 = createAudio(truth[:3000, :], audiosr=audiosr, winLength=winLength, frameshift=frameshift)

    #sf,wave=wavfile.read('/Volumes/Samsung_T5/data/SingleWordProductionDutch/results/sub-01_orig_synthesized.wav')
    import sounddevice as sd
    fs=16000
    sd.play(pred2,fs)
    sd.stop()

