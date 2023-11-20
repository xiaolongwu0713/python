import argparse
import os
import random
import time
from datetime import datetime
from dotmap import DotMap
from comm_utils import *
# 3rd-Party Modules
import numpy as np
import torch

# Self-Written Modules

from gesture.DA.timegan.models.timegan import TimeGAN
from gesture.DA.timegan.models.utils import timegan_trainer, timegan_generator
from gesture.utils import read_channel_number


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#########################
# Load and preprocess data for model
#########################

def timegan(sid,dataloader,save_gen_dir,result_dir,classi,scaler,args):
    #X=np.zeros([32,500,208])
    #T=[500,]*32
    # Experiment Arguments
    args.device = 'cuda'
    # args.exp='test'
    args.is_train = True
    args.seed = 42
    args.feat_prod_no = 1
    args.max_seq_len = 500
    args.train_rate = 0.5
    args.hidden_dim = 100
    args.num_layers = 10
    args.learning_rate = 0.0001
    args.optimizer = 'adam'
    args.dis_thresh = 0.00015  # 0.15 # update discriminator network if loss > dis_thresh
    args.padding_value = 0
    args.tensorboard_path = result_dir + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if running_from_CMD:
        args.emb_epochs = 200
        args.sup_epochs = 200
        args.joint_epochs = 400
    else:
        args.emb_epochs = 1
        args.sup_epochs = 1
        args.joint_epochs = 1

    args.gen_trials = 300
    args.save_gen_data = True

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        args.device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")
    chn_num=read_channel_number()
    one_batch = next(iter(dataloader))[0]
    args.model_path = result_dir
    args.result_dir = result_dir
    args.classi = classi
    args.batch_size = one_batch.shape[0]
    args.max_seq_len = one_batch.shape[1]
    args.wind = one_batch.shape[1]
    chn_num = one_batch.shape[2]
    args.feature_dim = chn_num
    args.Z_dim = chn_num

    args.losses={}
    # Log start time
    start = time.time()

    model = TimeGAN(args)

    if args.is_train == True:
        timegan_trainer(model, dataloader, args)
        filename = args.result_dir + 'losses_class_' + str(args.classi) + '.npy'
        np.save(filename, args.losses)
    end = time.time()
    print(f"Model Runtime: {(end - start) / 60} mins\n")

    ## generate data
    gen_data_tmp = timegan_generator(model, args) # (batch_size, wind, chn_number)
    gen_data_tmp=gen_data_tmp.transpose(0,2,1)
    print(gen_data_tmp.shape)

    if args.save_gen_data:
        if not args.test:
            #### scaling back ####
            gen_data = np.zeros((gen_data_tmp.shape))
            for i, trial in enumerate(gen_data_tmp):
                tmp = scaler.inverse_transform(trial.transpose())
                gen_data[i] = np.transpose(tmp)
            #### scaling back ####
        else: # test the sine generation
            gen_data=gen_data_tmp
        print("Saving generated data of class " + str(classi) + ".")
        if classi == 0:
            np.save(save_gen_dir + 'gen_class_0.npy', gen_data)
        elif classi == 1:
            np.save(save_gen_dir + 'gen_class_1.npy', gen_data)
        elif classi == 2:
            np.save(save_gen_dir + 'gen_class_2.npy', gen_data)
        elif classi == 3:
            np.save(save_gen_dir + 'gen_class_3.npy', gen_data)
        elif classi == 4:
            np.save(save_gen_dir + 'gen_class_4.npy', gen_data)

