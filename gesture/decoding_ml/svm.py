import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
from gesture.config import *

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from common_dl import set_random_seeds
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

#python svm.py $sid
if len(sys.argv)>1:
    sid = int(float(sys.argv[1]))
else: # debug in IDE
    sid=10

class_number=5
Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]
fs=1000

sid_dir=data_dir+'preprocessing'+'/P'+str(sid)+'/'
result_dir=data_dir+'training_result/machineLearning/P'+str(sid)+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

feature_file=sid_dir+'psd_feature.npy'
label_file=sid_dir+'label.npy'
list_of_epochs_psd_avg=np.load(feature_file)
list_of_labes=np.load(label_file)
print("Read data done./")
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
sss.get_n_splits(list_of_epochs_psd_avg, list_of_labes)
print("Feature selection./")
# use default SVM parameter--select feature--fine tune/gridsearch the SVM
feature_selection=True
if feature_selection:
    # initiate the clf with parameter calculated from gridsearch
    #svc_clf = SVC(kernel="linear",gamma='auto')
    svc_clf = LinearSVC(random_state=0, tol=1e-5)

    min_features_to_select = 5  # Minimum number of features to consider
    selector = RFECV(estimator=svc_clf, step=1, cv=sss,
                  scoring='accuracy', n_jobs=-1,
                  min_features_to_select=min_features_to_select)
    selec_pipe=make_pipeline(StandardScaler(), selector)
    selec_pipe.fit(list_of_epochs_psd_avg, list_of_labes)
    #selector.fit(list_of_epochs_psd_avg, list_of_labes)
    print("Optimal number of features : %d" % selector.n_features_) #feature number corresponds to highest accuracy

    filename = result_dir + 'SVM_accVSfeat'
    np.save(filename,np.asarray(selector.grid_scores_))

    ranks=selector.ranking_
    filename = result_dir + 'SVM_feature_rank'
    np.save(filename, np.asarray(ranks))
    print('Feature selection done.')

grid_search=False
if grid_search:
    # can't insert into a pipeline.
    #pipe_svc = make_pipeline(selector, StandardScaler(), SVC(kernel="linear",gamma='auto'))
    pipe_svc = make_pipeline(StandardScaler(), SVC(kernel="linear", gamma='auto'))
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    c_range=np.arange(0.00005,0.0002,0.00001)
    g_range=np.arange(0.0005,0.002,0.0001)
    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': c_range,
                   'svc__gamma': g_range,
                   'svc__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      refit=True,
                      #cv=8,
                      cv=sss,
                      n_jobs=-1)

    gs = gs.fit(list_of_epochs_psd_avg, list_of_labes)
    print(gs.best_score_)
    print(gs.best_params_)
    clf = gs.best_estimator_
    print('Test accuracy: %.3f' % clf.score(xtest, ytest))


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





