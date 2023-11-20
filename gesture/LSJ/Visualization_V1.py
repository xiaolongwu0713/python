
from EEGModels import DeepConvNet_210519_512_10_SpaFst_Dpth
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.model_selection import StratifiedKFold
import os
import h5py
from tensorflow.keras.models import load_model, Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# def model_load(subj, ):
#     classNum = 5
#
#     kfolds = 10
#     epochsCt = 50
#
#     pn = subj
#     loadPath = 'H:/lsj/preprocessing_data/P' + str(pn) + '/preprocessing3_Algorithm/preprocessingALL_3_Algorithm_v3.mat'
#     matDict = h5py.File(loadPath, 'r')
#     data = matDict['preData']
#     data = np.transpose(data, (4, 3, 2, 1, 0))
#     label = matDict['preLabel']
#     label = np.transpose(label, (1, 0))
#     label = label.astype('int64')
#     tampLbl = label[:, 1]
#     trial, strideInx, kernel, channel, sample = data.shape
#     Inx = np.arange(trial)
#
#     Accu  =  0
#
#     tkFold = StratifiedKFold(n_splits = kfolds, shuffle=True)
#     for i, j in tkFold.split(Inx, tampLbl):
#
#         trainData, testData = data[i], data[j]
#         trainLabel, testLabel = label[i], label[j]
#         trainData = np.reshape(trainData, (-1, kernel, channel, sample ))
#         testData = np.reshape(testData, (-1, kernel, channel, sample ))
#         trainLabel = np.reshape(trainLabel, (-1, 1))
#         testLabel = np.reshape(testLabel, (-1, 1))
#         trainLabel = np.eye(classNum)[trainLabel - 1]
#         testLabel = np.eye(classNum)[testLabel - 1]
#         trainLabel = np.reshape(trainLabel, [-1, classNum])
#         testLabel = np.reshape(testLabel, [-1, classNum])
#
#         model = DeepConvNet_210519_512_10_SpaFst_Dpth(classNum, Chans=channel, Samples=sample,
#                             dropoutRate=0.5)
#         model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
#
#         varsName = 'D:/lsj/Modelvari_CNN/P' + str(pn) + '/Vis_Model_NoTr_temp_0916.h5'    ### V2_P10_0909, 纠正之前 .mat没加的错误
#                                                                                           ### V2_p10_0908, 使用的Sig是re-reference的
#                                                                                           ### V3_P10_0915, 使用的Sig是No re-reference的
#                                                                                           ### V3_P41_0916, 使用的Sig是re-reference的
#         model.save(varsName)
#         model.fit(trainData, trainLabel, batch_size=32, epochs=10)
#         for epoch in range(epochsCt):
#             model.fit(trainData, trainLabel, batch_size= 32,  epochs= 2)
#             los, accu = model.evaluate(testData, testLabel)
#             if accu > Accu:
#                 MO = model
#                 TrData = trainData
#                 TrLabel = trainLabel
#                 TsData = testData
#                 TsLabel  =  testLabel
#                 Accu = accu
#                 TrInx = i
#                 TsInx = j
#                 MODEL_PATH = 'D:/lsj/Modelvari_CNN/P' + str(pn) + '/Vis_Model_NoTr_temp_0916.h5'
#                 MOnoTr  =  load_model(MODEL_PATH)
#                 varsName = 'D:/lsj/Modelvari_CNN/P' + str(pn) + '/Vis_Model_NoTr_0916.h5'
#                 MOnoTr.save(varsName)
#
#     varsName = 'D:/lsj/Modelvari_CNN/P' + str(pn) + '/Vis_Data_0916.mat'
#     savemat(varsName, {'trainData': TrData, 'trainLabel' : TrLabel, 'testData': TsData, 'testLabel' :TsLabel , 'trainInx': TrInx, 'testInx': TsInx} )
#     varsName = 'D:/lsj/Modelvari_CNN/P' + str(pn) + '/Vis_Model_0916.h5'
#     MO.save(varsName)
#
#
# Inf = [[2, 1000], [3, 1000], [ 4, 1000], [5, 1000], [ 7, 1000], [ 8, 1000], [ 9, 1000], [ 10, 2000],
#     [13, 2000], [ 16, 2000], [ 17, 2000], [ 18, 2000], [ 19, 2000], [ 20, 1000], [ 21, 1000], [ 22, 2000], [ 23, 2000],  #[24, 2000], [ 25, 2000], [ 26, 2000],
#      [  29, 2000], [ 30, 2000], [ 31, 2000], [ 32, 2000], [ 34, 2000], [ 35, 1000],
#     [36, 2000], [ 37, 2000], [41, 2000], [45, 2000]]
# Inf = np.array(Inf)
# Inf = Inf[[0,1,2,7,8,11,15,17,20,21,25,26],:]
#
# subjNum = np.size(Inf, 0)
# for subj in range( subjNum ):
#     Pn  = Inf[subj, 0]
#     model_load(Pn, )
#
#     """
#     0908 - BSSCNet, Sig重参考过, p10
#     0915 - BSSCNet, Sig未重参考
#     0916 - 同 0908, p剩余
#     """
#     # classNum = 5
#     # pn  = 41
#     # loadPath = 'H:/lsj/preprocessing_data/P' + str(pn) + '/preprocessing3_Algorithm/preprocessingALL_3_Algorithm_v3.mat'
#     # matDict = h5py.File(loadPath, 'r')
#     # data = matDict['preData']
#     # data = np.transpose(data, (4, 3, 2, 1, 0))
#     # label = matDict['preLabel']
#     # label = np.transpose(label, (1, 0))
#     # label = label.astype('int64')
#     # tampLbl = label[:, 1]
#     # trial, strideInx, kernel, channel, sample = data.shape
#     # Inx = np.arange(trial)
#     #
#     # Data = np.reshape(data, (-1, kernel, channel, sample ))
#     #
#     # Label = np.reshape(label, (-1, 1))
#     # Label = np.eye(classNum)[Label - 1]
#     # Label = np.reshape(Label, [-1, classNum])
#
#     MODEL_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Vis_Model_0916.h5'
#     model  =  load_model(MODEL_PATH)
#
#     VAR_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Vis_Data_0916.mat'
#     Dict = loadmat(VAR_PATH)
#     testData = Dict['testData']
#
#
#     weight_spati, bias_spati = model.get_layer('conv').get_weights()
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F1_weight_0916.mat'
#     savemat(SAVE_PATH, {'weight_spati': weight_spati, 'bias_spati': bias_spati })
#
#     weight_spati = model.get_layer('depth').get_weights()
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F4_weight_0916.mat'
#     savemat(SAVE_PATH, {'weight_spati': weight_spati, 'bias_spati': bias_spati })
#
#     weight_spati, bias_spati = model.get_layer('den').get_weights()
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F22_weight_0916.mat'
#     savemat(SAVE_PATH, {'weight_spati': weight_spati, 'bias_spati': bias_spati })
#
#
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[0].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F0_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F0': features_root})
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[2].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F2_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F2': features_root})
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[3].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F3_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F3': features_root})
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[4].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F4_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F4': features_root})
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[7].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F7_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F7': features_root})
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[11].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F11_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F11': features_root})
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[15].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F15_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F15': features_root})
#
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[19].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F19_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F19': features_root})
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[21].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F21_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F21': features_root})
#
#     layers_root = Model(inputs=model.input, outputs=model.layers[23].output)
#     features_root = layers_root.predict(testData)
#     SAVE_PATH = 'D:/lsj/Modelvari_CNN/P' + str(Pn) + '/Model_F23_output_0916.mat'
#     savemat(SAVE_PATH, {'features_root_F23': features_root})


Pn = 10
MODEL_PATH = 'D:/lsj/Modelvari_CNN/Model_Visualization_V2_p10_0908.h5'
model  =  load_model(MODEL_PATH)

VAR_PATH = 'D:/lsj/Modelvari_CNN/Visualization_V2_p10_0908.mat'
Dict = loadmat(VAR_PATH)
testData = Dict['testData']

layers_root = Model(inputs=model.input, outputs=model.layers[4].output)
features_root = layers_root.predict(testData)
SAVE_PATH = 'D:/lsj/Modelvari_CNN/Model_Visual_V2_F4_p10_0908.mat'
savemat(SAVE_PATH, {'features_root_F4': features_root})



















