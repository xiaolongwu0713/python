#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt

from comm_utils import slide_epochs
from common_dl import myDataset
from gesture.utils import read_data
from gesture.config import *

from example.grad_cam_pytorch.grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False



def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy() # (224, 224)/ (208, 500)
    cmap = cm.jet_r(gcam)[..., :3] * 255.0 # numpy.ndarray: (224, 224, 3)
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)



"""
Generate Grad-CAM at different layers of ResNet-152
"""
cuda=True
device = get_device(cuda)

sid=10
fs=1000
wind=500
stride=100
retrain_use_selected_channel=False
test_epochs, val_epochs, train_epochs = read_data(sid, fs, retrain_use_selected_channel)
X_test = []
y_test = []
for clas, epochi in enumerate(test_epochs):
    Xi, y = slide_epochs(epochi, clas, wind, stride)
    assert Xi.shape[0] == len(y)
    X_test.append(Xi)
    y_test.append(y)
X_test = np.concatenate(X_test, axis=0)  # (1300, 63, 500)
y_test = np.asarray(y_test)
y_test = np.reshape(y_test, (-1, 1))  # (5, 270)
test_set = myDataset(X_test, y_test)
batch_size = 1
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)
x_batch, y_batch = iter(test_loader).next()
x,y=x_batch[0],y_batch[0]

# Model
savepath = '/Users/long/My Drive/python/gesture/result/deepLearning/10/checkpoint_deepnet_33.pth'
checkpoint = torch.load(savepath,map_location=torch.device('cpu'))
from gesture.models.deepmodel import deepnet
n_chans=208
class_number=5
wind=500
model = deepnet(n_chans, class_number, wind)  # 81%
model.load_state_dict(checkpoint['net'])
fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [4, 1]})
fig.tight_layout()
#axi.set_xticklabels([])
#axi.set_xticks([])

model.to(device)
model.eval()
target_layers = ["conv_time"]
gcam = GradCAM(model=model)
target_class = 1
probs, ids = gcam.forward(x_batch.float())
ids_ = torch.LongTensor([[target_class]] * 1).to(device)
gcam.backward(ids=ids_)

for target_layer in target_layers:
    ax0.clear()
    regions = gcam.generate(target_layer=target_layer)
    regions = regions.cpu().numpy()  # (224, 224)/ (208, 500)
    im=ax0.imshow(np.squeeze(regions), origin='lower', cmap='RdBu_r')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.05, 0.3, 0.03, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    filename = result_dir + 'deepLearning/'+str(sid)+'/grad_cam_random_'+str(target_class)+'.pdf'
    fig.savefig(filename)

regions=np.squeeze(regions)
grad_channel=np.average(regions, axis=1)
ax1.plot(grad_channel, range(len(grad_channel)))



