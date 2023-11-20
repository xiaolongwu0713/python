## By using a customized 4-class data points, it achieves very high classification accuracy. It can select all informative features.
# This code doesn't need to pre-define the number of feature to be selected.

import sys
from stg import STG
import numpy as np
import scipy.stats # for creating a simple dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataset import create_twomoon_dataset,create_4lines_dataset
import torch

n_size = 1000 #Number of samples
p_size = 20   #Number of features
X_data, y_data=create_twomoon_dataset(n_size,p_size)
# use a customized dataset which contains 4 classes, 2 relative and 18 un-relateve features
X_data, y_data=create_4lines_dataset(2000,p_size)


# f, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].scatter(x=X_data[:, 0], y=X_data[:, 1], s=150, c=y_data.reshape(-1), alpha=0.4, cmap=plt.cm.get_cmap('RdYlBu'), )
# ax[0].set_xlabel('$x_1$', fontsize=20)
# ax[0].set_ylabel('$x_2$', fontsize=20)
# ax[0].set_title('Target y')
# ax[1].scatter(x=X_data[:, 2], y=X_data[:, 3], s=150, c=y_data.reshape(-1), alpha=0.4, cmap=plt.cm.get_cmap('RdYlBu'), )
# ax[1].set_xlabel('$x_3$', fontsize=20)
# ax[1].set_ylabel('$x_4$', fontsize=20)
# ax[1].set_title('Target y')
# plt.tick_params(labelsize=10)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

args_cuda = torch.cuda.is_available()
device = torch.device("cuda" if args_cuda else "cpu").type
feature_selection = True
output_dim=4
model = STG(task_type='classification',input_dim=X_train.shape[1], output_dim=output_dim, hidden_dims=[60, 20], activation='tanh',
    optimizer='SGD', learning_rate=0.1, batch_size=X_train.shape[0], feature_selection=feature_selection, sigma=0.5, lam=0.5, random_state=1, device=device)

model.fit(X_train, y_train, nr_epochs=60000, valid_X=X_valid, valid_y=y_valid, print_interval=1000)

model.get_gates(mode='prob')
model.get_gates(mode='raw')

# testing
y_pred=model.predict(X_data)
y_pred[:10]
y_data[:10]

f, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(x=X_data[:, 0], y=X_data[:, 1], s=150, c=y_data.reshape(-1), alpha=0.4, cmap=plt.cm.get_cmap('RdYlBu'), )
ax[0].set_xlabel('$x_1$', fontsize=20)
ax[0].set_ylabel('$x_2$', fontsize=20)
ax[0].set_title('Target y')
ax[1].scatter(x=X_data[:, 0], y=X_data[:, 1], s=150, c=y_pred.reshape(-1), alpha=0.4, cmap=plt.cm.get_cmap('RdYlBu'), )
ax[1].set_xlabel('$x_1$', fontsize=20)
ax[1].set_ylabel('$x_2$', fontsize=20)
ax[1].set_title('Classification output ')
plt.tick_params(labelsize=10)





