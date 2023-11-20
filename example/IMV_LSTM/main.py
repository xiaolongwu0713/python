import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from networks import IMVTensorLSTM
import time
data1 = pd.read_csv("SML2010/NEW-DATA-1.T15.txt", sep=' ')
data2 = pd.read_csv("SML2010/NEW-DATA-2.T15.txt", sep=' ')
target = '3:Temperature_Comedor_Sensor'
cols = [
 '4:Temperature_Habitacion_Sensor',
 '5:Weather_Temperature',
 '6:CO2_Comedor_Sensor',
 '7:CO2_Habitacion_Sensor',
 '8:Humedad_Comedor_Sensor',
 '9:Humedad_Habitacion_Sensor',
 '10:Lighting_Comedor_Sensor',
 '11:Lighting_Habitacion_Sensor',
 '12:Precipitacion',
 '13:Meteo_Exterior_Crepusculo',
 '14:Meteo_Exterior_Viento',
 '15:Meteo_Exterior_Sol_Oest',
 '16:Meteo_Exterior_Sol_Est',
 '20:Exterior_Entalpic_2',
 '21:Exterior_Entalpic_turbo',
 '22:Temperature_Exterior_Sensor']

train_size = 3200
val_size = 400
depth = 10
batch_size = 128
prediction_horizon = 1
X_train1 = np.zeros((len(data1), depth, len(cols)))
yy_train1 = np.zeros((len(data1), 1))
for i, name in enumerate(cols):
    for j in range(depth):
        X_train1[:, j, i] = data1[name].shift(depth - j - 1).fillna(method="bfill")
yy_train1 = data1[target].shift(-prediction_horizon).fillna(method='ffill')
X_train1 = X_train1[depth:-prediction_horizon]
yy_train1 = yy_train1[depth:-prediction_horizon]
X2 = np.zeros((len(data2), depth, len(cols)))
yy2 = np.zeros((len(data2), 1))
for i, name in enumerate(cols):
    for j in range(depth):
        X2[:, j, i] = data2[name].shift(depth - j - 1).fillna(method="bfill")
yy2 = data2[target].shift(-prediction_horizon).fillna(method='ffill')

X_train2 = X2[:train_size - len(data1)]
yy_train2 = yy2[:train_size - len(data1)]

X_val = X2[train_size - len(data1):train_size - len(data1) + val_size]
yy_val = yy2[train_size - len(data1):train_size - len(data1) + val_size]

X_test = X2[train_size - len(data1) + val_size:]
yy_test = yy2[train_size - len(data1) + val_size:]

X_train2 = X_train2[depth:]
yy_train2 = yy_train2[depth:]
X_train = np.concatenate([X_train1, X_train2], axis=0)
yy_train = np.concatenate([yy_train1, yy_train2], axis=0)
X_train_min, yy_train_min = X_train.min(axis=0), yy_train.min(axis=0)
X_train_max, yy_train_max = X_train.max(axis=0), yy_train.max(axis=0)
X_train = (X_train - X_train_min)/(X_train_max - X_train_min + 1e-9)
X_val = (X_val - X_train_min)/(X_train_max - X_train_min + 1e-9)
X_test = (X_test - X_train_min)/(X_train_max - X_train_min + 1e-9)

yy_train = (yy_train - yy_train_min)/(yy_train_max - yy_train_min + 1e-9)
yy_val = (yy_val - yy_train_min)/(yy_train_max - yy_train_min + 1e-9)
yy_test = (yy_test - yy_train_min)/(yy_train_max - yy_train_min + 1e-9)
X_train_t = torch.Tensor(X_train)
X_val_t = torch.Tensor(X_val)
X_test_t = torch.Tensor(X_test)
yy_train_t = torch.Tensor(yy_train)
yy_val_t = torch.Tensor(yy_val.values)
yy_test_t = torch.Tensor(yy_test.values)

# X_train_t:(batch, time, variables)
train_loader = DataLoader(TensorDataset(X_train_t, yy_train_t), shuffle=True, batch_size=batch_size)
val_loader = DataLoader(TensorDataset(X_val_t, yy_val_t), shuffle=False, batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test_t, yy_test_t), shuffle=False, batch_size=batch_size)

#model = IMVTensorLSTM(X_train.shape[2], 1, 128).cuda()
model = IMVTensorLSTM(X_train.shape[2], 1, 128)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)
from sklearn.metrics import mean_squared_error, mean_absolute_error

epochs = 12
loss = nn.MSELoss()
patience = 35
min_val_loss = 9999
counter = 0
for i in range(epochs):
    print("Epoch " + str(i))
    mse_train = 0
    iteration_start = time.monotonic()
    for batch_x, batch_yy in train_loader:
        # batch_x = batch_x.cuda()
        # batch_y = batch_y.cuda()
        batch_x = batch_x
        batch_yy = batch_yy
        opt.zero_grad()
        y_pred, alphas, betas = model(batch_x)
        y_pred = y_pred.squeeze(1)
        l = loss(y_pred, batch_yy)
        l.backward()
        mse_train += l.item() * batch_x.shape[0]
        opt.step()
    epoch_scheduler.step()
    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y in val_loader:
            # batch_x = batch_x.cuda()
            # batch_y = batch_y.cuda()
            batch_x = batch_x
            batch_y = batch_y
            output, alphas, betas = model(batch_x)
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(output, batch_y).item() * batch_x.shape[0]
    preds = np.concatenate(preds)
    true = np.concatenate(true)

    if min_val_loss > mse_val ** 0.5:
        min_val_loss = mse_val ** 0.5
        print("Saving...")
        torch.save(model.state_dict(), "imv_lstm_sml2010.pt")
        counter = 0
    else:
        counter += 1

    if counter == patience:
        break
    print("Iter: ", i, "train: ", (mse_train / len(X_train_t)) ** 0.5, "val: ", (mse_val / len(X_val_t)) ** 0.5)
    iteration_end = time.monotonic()
    print("Iter time: ", iteration_end - iteration_start)
    if (i % 10 == 0):
        preds = preds * (yy_train_max - yy_train_min) + yy_train_min
        true = true * (yy_train_max - yy_train_min) + yy_train_min
        mse = mean_squared_error(true, preds)
        mae = mean_absolute_error(true, preds)
        print("mse: ", mse, "mae: ", mae)
        plt.figure(figsize=(20, 10))
        plt.plot(preds)
        plt.plot(true)
        plt.show()

with torch.no_grad():
    mse_val = 0
    preds = []
    true = []
    alphas = []
    betas = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x
        batch_y = batch_y
        output, a, b = model(batch_x)
        output = output.squeeze(1)
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
        alphas.append(a.detach().cpu().numpy())
        betas.append(b.detach().cpu().numpy())
        mse_val += loss(output, batch_y).item()*batch_x.shape[0]
preds = np.concatenate(preds)
true = np.concatenate(true)
alphas = np.concatenate(alphas)
betas = np.concatenate(betas)

preds = preds*(yy_train_max - yy_train_min) + yy_train_min
true = true*(yy_train_max - yy_train_min) + yy_train_min
plt.figure(figsize=(20, 10))
plt.plot(preds)
plt.plot(true)
plt.show()






