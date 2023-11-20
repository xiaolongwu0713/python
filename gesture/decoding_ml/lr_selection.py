#%cd /content/drive/MyDrive/
# raw_data is imported from global config

#%%capture
#! pip install hdf5storage
#! pip install mne==0.23.0
#! pip install torch
#! pip install Braindecode==0.5.1
#! pip install timm

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from common_dl import set_random_seeds
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gesture.decoding_ml.genInput_psd import load_data_ml_psd
from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

#sys.stdout.write(i.getsource(deepnet))

sid=10 #4
class_number=5
#C=0.1
Session_num,UseChn,EmgChn,TrigChn, activeChan = get_channel_setting(sid)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]
fs=1000

project_dir=data_dir+'preprocessing'+'/P'+str(sid)+'/'
selection_dir=project_dir + 'selection/lr/'
if not os.path.exists(selection_dir):
    os.makedirs(selection_dir)

list_of_epochs_psd_avg,list_of_labes=load_data_ml_psd(10,channel='active')
X_train,X_test,y_train,y_test=train_test_split(list_of_epochs_psd_avg,list_of_labes,test_size=0.4,random_state=0)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
sss.get_n_splits(X_train, y_train)

l1_ratio = 0.5  # L1 weight in the Elastic-Net regularization
feature_selection=True
penalties=["L1 penalty","Elastic-Net","L2 penalty"]
if feature_selection:
    fig, ax = plt.subplots()
    #for i, (C, axes_row) in enumerate(zip((1, 0.1, 0.01), axes)):
    for C in [1,0.1,0.01]:
        clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga')
        clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01, solver='saga')
        clf_en_LR = LogisticRegression(C=C, penalty='elasticnet', solver='saga',l1_ratio=l1_ratio, tol=0.01)

        clf_l1_LR = Pipeline([('scaler',StandardScaler()), ('clf',clf_l1_LR)])
        clf_l2_LR = Pipeline([('scaler',StandardScaler()), ('clf',clf_l2_LR)])
        clf_en_LR = Pipeline([('scaler',StandardScaler()), ('clf',clf_en_LR)])

        clf_l1_LR.fit(X_train, y_train)
        clf_l2_LR.fit(X_train, y_train)
        clf_en_LR.fit(X_train, y_train)

        coef_l1_LR = clf_l1_LR.named_steps['clf'].coef_.ravel()
        coef_l2_LR = clf_l2_LR.named_steps['clf'].coef_.ravel()
        coef_en_LR = clf_en_LR.named_steps['clf'].coef_.ravel()

        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
        sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
        sparsity_en_LR = np.mean(coef_en_LR == 0) * 100

        print("C=%.2f" % C)
        print("{:<40} {:.2f}%".format("Sparsity with L1 penalty:", sparsity_l1_LR))
        print("{:<40} {:.2f}%".format("Sparsity with Elastic-Net penalty:",sparsity_en_LR))
        print("{:<40} {:.2f}%".format("Sparsity with L2 penalty:", sparsity_l2_LR))

        print("{:<40} {:.2f}".format("Score with L1 penalty:",clf_l1_LR.score(X_test, y_test)))
        print("{:<40} {:.2f}".format("Score with Elastic-Net penalty:",clf_en_LR.score(X_test, y_test)))
        print("{:<40} {:.2f}".format("Score with L2 penalty:",clf_l2_LR.score(X_test, y_test)))

        for i,coefs in enumerate([coef_l1_LR, coef_en_LR, coef_l2_LR]):
            #ax.imshow(np.abs(coefs.reshape(207, 5)), interpolation='nearest',cmap='binary') #, vmax=1, vmin=0
            ax.plot(coefs)
            filename = selection_dir + 'LogReg_C' + str(C)+ '_'+penalties[i] + '.pdf'
            fig.savefig(filename)
            ax.clear()


train_only=False
if train_only:
    accuracy = []
    for train_index, test_index in sss.split(list_of_epochs_psd_avg, list_of_labes):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = list_of_epochs_psd_avg[train_index], list_of_epochs_psd_avg[test_index]
        y_train, y_test = list_of_labes[train_index], list_of_labes[test_index]
        # clf = make_pipeline(SVC(gamma='auto'))
        # input format: X:(samples, feature_number) y:(samples,)
        pipe_svc_best.fit(X_train, y_train)
        y_predict = pipe_svc_best.predict(X_test)
        confusion_matrix(y_test, y_predict)
        accu = sum(y_test == y_predict) / len(y_test)
        accuracy.append(accu)
        print("Accuracy: %.2f" % (accu))
    accuracy = np.asarray(accuracy)
    filename = selection_dir + 'accuracy_sid' + str(sid)
    np.save(filename, accuracy)





