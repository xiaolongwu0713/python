import pandas as pd
import numpy as np
import pyunpack
import math
import json

from data.data_download import Config, download_electricity
from data_formatters.electricity import ElectricityFormatter
from data_formatters.base import DataTypes, InputTypes

from data.custom_dataset import TFTDataset
from models import GatedLinearUnit
from models import GateAddNormNetwork
from models import GatedResidualNetwork
from models import ScaledDotProductAttention
from models import InterpretableMultiHeadAttention
from models import VariableSelectionNetwork

from quantile_loss import QuantileLossCalculator
from quantile_loss import NormalizedQuantileLossCalculator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from argparse import ArgumentParser
from TFT import TemporalFusionTransformer

import matplotlib.pyplot as plt

config = Config('data','data/electricity.csv')
#download_electricity(config)

electricity = pd.read_csv('data/electricity.csv', index_col = 0)
data_formatter = ElectricityFormatter()
train, valid, test = data_formatter.split_data(electricity)

train.days_from_start.value_counts().to_frame().reset_index().sort_values(by=['index'])
valid.days_from_start.value_counts().to_frame().reset_index().sort_values(by=['index'])
test.days_from_start.value_counts().to_frame().reset_index().sort_values(by=['index'])

test = test.reset_index(drop=True)
test[test.categorical_id == 0]
test.groupby(['categorical_id']).apply(lambda x: x.shape[0]).mean()
g = test.groupby(['categorical_id'])
data_formatter.get_time_steps()

df_index_abs = g[['categorical_id']].transform(lambda x: x.index+data_formatter.get_time_steps()) \
                        .reset_index() \
                        .rename(columns={'index':'init_abs',
                                         'categorical_id':'end_abs'})
df_index_rel_init = g[['categorical_id']].transform(lambda x: x.reset_index(drop=True).index) \
                        .rename(columns={'categorical_id':'init_rel'})
df_index_rel_end = g[['categorical_id']].transform(lambda x: x.reset_index(drop=True).index+data_formatter.get_time_steps()) \
                .rename(columns={'categorical_id':'end_rel'})
df_total_count = g[['categorical_id']].transform(lambda x: x.shape[0] - data_formatter.get_time_steps() + 1) \
                .rename(columns = {'categorical_id':'group_count'})
new_test = pd.concat([df_index_abs,
                       df_index_rel_init,
                       df_index_rel_end,
                       test[['id']],
                       df_total_count], axis = 1).reset_index(drop = True)
new_test[new_test.end_rel < test.groupby(['categorical_id']).apply(lambda x: x.shape[0]).mean()].reset_index()
train_dataset = TFTDataset(train)
valid_dataset = TFTDataset(valid)
test_dataset = TFTDataset(test)

params = data_formatter.get_experiment_params()
params.update(data_formatter.get_default_model_params())

parser = ArgumentParser(add_help=False)
for k in params:
    if type(params[k]) in [int, float]:
        #if k == 'minibatch_size':
        #    parser.add_argument('--{}'.format(k), type=type(params[k]), default = 256)
        #else:
        parser.add_argument('--{}'.format(k), type=type(params[k]), default = params[k])
    else:
        parser.add_argument('--{}'.format(k), type=str, default = str(params[k]))
hparams = parser.parse_known_args()[0]

tft = TemporalFusionTransformer(hparams,train_dataset,valid_dataset,test_dataset)#.to(DEVICE)

early_stop_callback = EarlyStopping(monitor = 'val_loss',
                                    min_delta = 1e-4,
                                    patience = tft.early_stopping_patience,
                                    verbose=False,
                                    mode='min')
trainer = pl.Trainer(max_nb_epochs = tft.num_epochs,
                     gpus = 0,
                     track_grad_norm = 2,
                     gradient_clip_val = tft.max_gradient_norm,
                     early_stop_callback = early_stop_callback,
                     #train_percent_check = 0.01,
                     #val_percent_check = 0.01,
                     #test_percent_check = 0.01,
                     overfit_pct=0.01,
                     #fast_dev_run=True,
                     profiler=True,
                     #print_nan_grads = True,
                     #distributed_backend='dp'
                    )
trainer.fit(tft)

trainer.test()

q_risk = NormalizedQuantileLossCalculator([0.1, 0.5, 0.9], 1)
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=False, drop_last=True)

loss = []
batches = 0
for i, (batch, target, _ )in enumerate(test_dataloader):
    if i < 5:
        t = target
        batches += 1
        output = tft(batch)
        loss.append(q_risk.apply(output[Ellipsis, 1], target[Ellipsis, 0], 0.5))
    else:
        break
mean_loss = sum(loss) / batches
mean_loss

def get_decoder_mask(self_attn_inputs):
    """Returns causal mask to apply for self-attention layer.
    Args:
    self_attn_inputs: Inputs to self attention layer to determine mask shape
    """
    len_s = self_attn_inputs.shape[1]
    bs = self_attn_inputs.shape[0]
    mask = torch.cumsum(torch.eye(len_s), 0)
    mask = mask.repeat(bs,1,1).to(torch.float32)

    return mask

a = torch.randn((2,6,4))
mask = get_decoder_mask(a)

linear = nn.Linear(4, 2, bias = False)
a_lin = linear(a)
to_attn = torch.bmm(a_lin, a_lin.permute(0,2,1))
masked_attn = to_attn.masked_fill(mask == 0, -1e9)
softmax = nn.Softmax(dim = 2)
sft_attn = softmax(masked_attn)
torch.bmm(sft_attn, a_lin).shape
scaled_att = ScaledDotProductAttention()
scaled_att(a_lin, a_lin, a_lin, mask = get_decoder_mask(a))

