import argparse
import sys, os
import socket

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/mydrive/python'])
elif socket.gethostname() == 'LongsMac':
    sys.path.extend(['/Users/long/My Drive/python'])
elif socket.gethostname() == 'DESKTOP-NP9A9VI':
    sys.path.extend(['C:/Users/xiaol/My Drive/python/'])
elif socket.gethostname() == 'Long': # Yoga
    sys.path.extend(['D:/mydrive/python/'])


from gesture.DA.VAE.VAE import SEEG_CNN_VAE, loss_fn, gen_data_vae
import PIL
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm
from gesture.DA.metrics.visualization import visualization
from gesture.DA.cTGAN.utils import save_checkpoint
import dateutil, pytz
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gesture.channel_selection.utils import get_good_sids, get_final_good_sids, get_selected_channel_gumbel
from gesture.utils import *
from gesture.config import *
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', default=10, type=int)
    parser.add_argument('--cv', type=int)
    parser.add_argument('--load_path',type=str,help='The reload model path')
    parser.add_argument('--epoch', type=int,default=200)
    opt = parser.parse_args()
    return opt

args = parse_args()

if running_from_CMD:
    sid = args.sid
    cv=args.cv
elif running_from_IDE:
    sid=10
    cv=1
fs=1000
wind=500
stride=100
gen_method='VAE'
epochs=args.epoch
selected_channels='Yes'

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True


now = datetime.now(pytz.timezone('Asia/Shanghai'))  # dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
prefix = tmp_dir + 'DA/VAE/sid' + str(args.sid) + '/cv'+str(cv)+'/' + timestamp + '/'
print("Result dir: "+prefix+".")
sample_path = prefix + 'Samples/'
log_path=prefix+'Log/'
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

writer = SummaryWriter(log_path)
#args = cfg_cmd.parse_args()
latent_dims = 512
#args.load_path='D:/tmp/python/gesture/DA/CWGANGP/10/2024_04_11_11_34_05/Model/checkpoint_290.pth'
# print(str(sid)+','+str(fs)+','+str(wind)+','+str(stride)+','+gen_method+','+continuous+','+str(epochs)+','+str(selected_channels))

class_number = 5
if selected_channels:
    selected_channels, acc = get_selected_channel_gumbel(args.sid, 10)  # 10 selected channels
else:
    selected_channels = None

# can try to use selected channels
norm_method = 'minmax'  # minmax/std

# def train(sid,continuous,train_loader,chn_num):

batch_size = 32
test_epochs, val_epochs, train_epochs, scaler = read_data_split_function(sid, fs, selected_channels=selected_channels,
                                                                         scaler=norm_method,cv_idx=args.cv)
X_train, y_train, X_val, y_val, X_test, y_test = windowed_data(train_epochs, val_epochs, test_epochs, wind, stride)
total_trials = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]

#X_train = np.concatenate((X_train, X_val), axis=0)
#y_train = np.concatenate((y_train, y_val))
train_set = myDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
one_batch = next(iter(train_loader))[0]  # torch.Size([32, 208, 500])
chn_num = one_batch.shape[1]
labels=np.array([i[0] for i in y_train.tolist()])
X_train_class0, labels0 = X_train[labels==0,:,:], labels[labels==0] # (304, 10, 500)
plot_true=X_train_class0.transpose(0, 2, 1)[:304, :, :]
scalers = {}
#plot_true=plot_true.transpose(0,2,1) # batch, time, channel
for i in range(plot_true.shape[0]):
    scalers[i] = StandardScaler() # time, channel
    plot_true[i, :, :] = scalers[i].fit_transform(plot_true[i,:, :])
#plot_true=plot_true.transpose(0,2,1)

model = SEEG_CNN_VAE(chn_num, class_number, wind).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#count_parameters(model)

print("Generate data using " + gen_method + ".")
#gan(gen_method, writer, sid, continuous, chn_num, class_number, wind, train_loader, epochs, latent_dims)
global_steps=0
fig,axs=plt.subplots(2,2)
plot_channel_num=2
for classi in range(5):
    print("Class" + str(classi)+ ".")
    for epoch in range(epochs):
        model.train()
        losss, recon_losss, klds=[],[],[]
        for i, data in enumerate(train_loader):

            x, y = data
            x, y = x.to(device), y.to(device)
            x = x.type(Tensor)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x) # torch.Size([5, 2, 1500])
            loss, recon_loss, kld = loss_fn(recon_x, x, mu, logvar)
            losss.append(loss)
            recon_losss.append(recon_loss)
            klds.append(kld)

            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss, global_steps)
            writer.add_scalar('recon_loss', recon_loss, global_steps)
            writer.add_scalar('KLD', kld, global_steps)
            global_steps+=1
        print(
            "Epoch {:3d}/{:3d} Loss({:.2f}) = Recon_loss({:.2f}) + KLD({:.2f}).".
            format(epoch, epochs, sum(losss)/len(losss), sum(recon_losss)/len(recon_losss),sum(klds)/len(klds)))
        # evaluate
        num_epochs_to_generate=304
        gen_data_tmp = gen_data_vae(model, num_epochs_to_generate)

        scalers = {}
        tmp=gen_data_tmp.transpose(0,2,1) # batch, time, channel
        for i in range(tmp.shape[0]):
            scalers[i] = StandardScaler()
            tmp[i, :, :] = scalers[i].fit_transform(tmp[i,:, :])
        gen_data_tmp=tmp.transpose(0,2,1)

        for i in range(2):  # 4 plots
            for j in range(2):
                axs[i, j].clear()
                for k in range(plot_channel_num):
                    axs[i, j].plot(gen_data_tmp[i * 2 + j, k, :])
        plt.figure(fig)
        plt.savefig('del_figure.png')
        image = PIL.Image.open('del_figure.png')
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image('Image/raw', image[0], global_steps)


        ff = visualization([gen_data_tmp.transpose(0, 2, 1), plot_true], 'tSNE', labels,
                           display=False, epoch=epoch)
        plt.figure(ff)
        plt.savefig('del_figure2.png')
        image2 = PIL.Image.open('del_figure2.png')
        image2 = ToTensor()(image2).unsqueeze(0)
        writer.add_image('Image/tSNE', image2[0], global_steps)

    print("Training completed.")

    #if gen_after_train:
    ## generate new data
    num_epochs_to_generate=304
    gen_data_tmp=gen_data_vae(model, num_epochs_to_generate) # (500, 208, 500)
    ''' generate in another way
    trials,labels=next(iter(test_loader))
    trial,label=trials[0],labels[0] # torch.Size([208, 500])
    trial=torch.unsqueeze(trial,0)
    if generative_model=='VAE':
        gen_trial, mu, logvar=model(trial.type(Tensor))
    trial, gen_trial=torch.squeeze(trial), torch.squeeze(gen_trial)
    '''

    scalers = {}
    tmp = gen_data_tmp.transpose(0, 2, 1)  # batch, time, channel
    for i in range(tmp.shape[0]):
        scalers[i] = StandardScaler()
        tmp[i, :, :] = scalers[i].fit_transform(tmp[i, :, :])
    gen_data = tmp.transpose(0, 2, 1)

    ## scaling back
    # gen_data=np.zeros((gen_data_tmp.shape))
    # for i, triall in enumerate(gen_data_tmp):
    #     tmp=scaler.inverse_transform(triall.transpose())
    #     gen_data[i]=np.transpose(tmp)
    #trial, gen_trial=trial.transpose(), gen_trial.transpose()
    # compare original vs generated data
    print("Saving generated data of class " + str(classi) + ".")
    if classi == 0:
        np.save(sample_path + 'class0_'+'cv'+str(args.cv)+'.npy', gen_data)
    elif classi == 1:
        np.save(sample_path + 'class1_'+'cv'+str(args.cv)+'.npy', gen_data)
    elif classi == 2:
        np.save(sample_path + 'class2_'+'cv'+str(args.cv)+'.npy', gen_data)
    elif classi == 3:
        np.save(sample_path + 'class3_' + 'cv' + str(args.cv) + '.npy', gen_data)
    elif classi == 4:
        np.save(sample_path + 'class4_' + 'cv' + str(args.cv) + '.npy', gen_data)

    #break # only train/gen the first class