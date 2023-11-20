## usage: 1: traing a particular GAN model; 2: find a proper model (epoch) from tensorboard plot;
## 3: generate fake data in this script; 4: call ./loop_main_all.sh in decoding_dl folder

from gesture.DA.tts_cgan_main.TransCGAN_model import Generator
#from example.tts_gan_main.GANModels import Generator
from dotmap import DotMap

from gesture.DA.GAN.gan import SEEG_CNN_Generator_10channels
from gesture.DA.GAN.WGAN_GP import gen_data_wgangp_
from gesture.feature_selection.utils import get_selected_channel_gumbel
from gesture.utils import read_good_sids, read_sids, read_gen_data, windowed_data, \
    read_data_split_function, noise_injection_epoch
from gesture.config import *

#### Meta info ####
sid=10
fs=1000
gen_epochs=None
wind=500
stride=200
classi=0 # test on data from class 0
good_sids=read_good_sids()
sids=read_sids()
da_result_dir='H:/Long/data/gesture/DA/'
plot_class0=[]
#### original data ####
# test_epochs, val_epochs, train_epochs, scaler=read_data(sid,fs,scaler='std')
# X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_epochs,val_epochs,test_epochs,wind,stride)
# ori_data_all=np.concatenate((X_train,X_val,X_test))
# ori_label_all=np.squeeze(np.concatenate((y_train,y_val,y_test))) # 380*5=1900
# ori_data=ori_data_all[ori_label_all==classi] # (380, 208, 500)
sid=10
use_these_chanels, acc = get_selected_channel_gumbel(sid, 10)
norm_method='std'
test_epochs, val_epochs, train_epochs, scaler=read_data_split_function(sid,fs,selected_channels=use_these_chanels,scaler=norm_method)
X_train,y_train,X_val,y_val,X_test,y_test=windowed_data(train_epochs,val_epochs,test_epochs,wind,stride)

labels=np.array([i[0] for i in y_train.tolist()])
X_train_class0, labels0 = X_train[labels==0,:,:], labels[labels==0]
plot_class0.append(X_train_class0)

#### generate fake data ####
#args = cfg_cmd.parse_args()
#latent_dims=args.latent_dims
gen_method='CWGANGP' # TTSCGAN/CWGANGP
save_gen_dir = data_dir + 'preprocessing/' + 'P' + str(sid)+'/'+gen_method+'/'
def gen_cwgangp(sid,c, epoch,filename,trial_number):
    latent_dims=512
    generator = SEEG_CNN_Generator_10channels(10,latent_dims)
    G=filename+'checkpoint_G_epochs_'+str(epoch)+'.pth'
    D=filename+'checkpoint_D_epochs_'+str(epoch)+'.pth'
    checkpoint = torch.load(G)
    generator.load_state_dict(checkpoint['net'])
    generator.eval()

    z = torch.randn(trial_number, latent_dims)
    gen_label = torch.ones((trial_number, 1)).type(LongTensor).to('cpu') * c
    gen_data = generator(z, gen_label)  # torch.Size([1, 208, 500])
    gen_data = gen_data.detach().numpy()  # (304, 10, 500)

    return gen_data

# generate for class 0
for c in range(1): # 5
    epoch=799
    trial_num=304
    filename = da_result_dir + gen_method + '/' + str(sid) + '/2022_11_23_11_42_26/'
    gen_data=gen_cwgangp(sid, c,epoch,filename,trial_num)
    np.save(save_gen_dir + 'gen_class_'+str(c)+'_' + str(epoch + 1) + '.npy', gen_data)

plot_class0.append(gen_data)

#### timeGAN generated data: not good at all ####
method='timeGAN'
gen_data_all = read_gen_data(sid,method,gen_epochs,'std')
gen_data=gen_data_all[classi] # (300, 208, 500)

#### Noise injection data ####
method='NI'
std_scale=0.05 # how much noise added
sid=10
#use_these_chanels, acc = get_selected_channel_gumbel(sid, 10)
#norm_method='std'
#test_epochs, val_epochs, train_epochs, scaler=read_data_split_function(sid,fs,selected_channels=use_these_chanels,scaler=norm_method)
train_epochs_NI=noise_injection_epoch(train_epochs,std_scale)
X_train_NI,y_train_NI,X_val_NI,y_val_NI,X_test_NI,y_test_NI=windowed_data(train_epochs_NI,val_epochs,test_epochs,wind,stride)
labels_NI=np.array([i[0] for i in y_train_NI.tolist()])
X_train_class0_NI, labels0_NI = X_train_NI[labels_NI==0,:,:], labels_NI[labels_NI==0]
plot_class0.append(X_train_class0_NI)


#### wgan_gp ####
gen_method='WGANGP'
critic='deepnet' # ‘resnet'/'deepnet'
sid=10
ch_num=208

from gesture.DA.GAN.WGAN_GP import SEEG_CNN_Generator2
from gesture.DA.GAN.WGAN_GP import latent_dims as WGANGP_lantent_dims
from gesture.config import trial_num
num_data_to_generate=trial_num[str(sid)]
generator = SEEG_CNN_Generator2(10,'std').to(device)
classi=4
epoch=400  #1130 [1918/1319/1279/1329/1599]
filename=r"C:\Users\wuxiaolong\mydrive\python\gesture\result\DA\WGAN_GP\10_good\checkpoint_G_epochs_"+str(epoch)+".pth"
checkpoint = torch.load(filename)
generator.load_state_dict(checkpoint['net'])
generator.eval()
save_gen_dir = data_dir + 'preprocessing/' + 'P' + str(sid)+'/'+gen_method+'/'
gen_data=gen_data_wgangp_(generator,WGANGP_lantent_dims,num_data_to_generate=num_data_to_generate) #(304, 208, 500)
if classi == 0:
    np.save(save_gen_dir + 'gen_class_0_' + str(epoch + 1) + '.npy', gen_data)
elif classi == 1:
    np.save(save_gen_dir + 'gen_class_1_' + str(epoch + 1) + '.npy', gen_data)
elif classi == 2:
    np.save(save_gen_dir + 'gen_class_2_' + str(epoch + 1) + '.npy', gen_data)
elif classi == 3:
    np.save(save_gen_dir + 'gen_class_3_' + str(epoch + 1) + '.npy', gen_data)
elif classi == 4:
    np.save(save_gen_dir + 'gen_class_4_' + str(epoch + 1) + '.npy', gen_data)

#### tts_gan ####
args = DotMap()
args.chn = 208
args.wind = wind
args.stride=stride
gen_net = Generator(args)
epoch=2051
filename=r"H:\Long\data\gesture\DA\TTS-GAN\10\2022_11_02_18_33_08\Model\checkpoint_"+str(epoch)+".pth"
checkpoint = torch.load(filename)
gen_net.load_state_dict(checkpoint['gen_state_dict'])
gen_net.eval()
gen_data = []
#trial_number=ori_data.shape[0]
trial_number=380
for i in range(trial_number):
    fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100)))
    fake_sigs = np.squeeze(gen_net(fake_noise).detach().numpy()) # (1, 208, 1, 500)
    gen_data.append(fake_sigs)
gen_data=np.asarray(gen_data) # (500, 208, 500)

#### tts_cgan ####
gen_method='TTSCGAN'
args = DotMap()
args.fs=1000
args.batch_size=1
args.latent_dim=128
args.wind = 500
args.stride=200
args.class_num = 5  # 5
args.selected_channels = True
if args.selected_channels:
    args.chn = 10  # 1
else:
    args.chn = 208
#gen_net = Generator_TTSCGAN(seq_len=args.wind, channels=args.chn, num_classes=args.class_num,latent_dim=args.latent_dim, data_embed_dim=10,  # latent_dim=100
#                            label_embed_dim=10, depth=3, num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5)


gen_net = Generator(seq_len=args.wind, channels=args.chn, num_classes=args.class_num,latent_dim=args.latent_dim, data_embed_dim=10,  # latent_dim=100
                            label_embed_dim=10, depth=3, num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5)

save_gen_dir = data_dir + 'preprocessing/' + 'P' + str(sid)+'/'+gen_method+'/'
if not os.path.exists(save_gen_dir):
    os.makedirs(save_gen_dir)
sid=10
if sid==10:
    epoch=335 # 11232
    filename='H:/Long/data/gesture/DA/TTSCGAN/'+str(sid)+'/2022_12_01_10_59_14/Model/checkpoint_'+str(epoch)+'.pth'
elif sid==4:
    epoch=475
    filename = 'H:/Long/data/gesture/DA/TTSCGAN/' + str(sid) + '/2022_11_18_21_57_58/Model/checkpoint_' + str(epoch) + '.pth'
elif sid==2:
    epoch=174
    filename = 'H:/Long/data/gesture/DA/TTSCGAN/' + str(sid) + '/2022_11_18_17_25_16/Model/checkpoint_' + str(epoch) + '.pth'
elif sid==13:
    epoch=337
    filename = 'H:/Long/data/gesture/DA/TTSCGAN/' + str(sid) + '/2022_11_19_11_23_46/Model/checkpoint_' + str(epoch) + '.pth'


checkpoint = torch.load(filename)
gen_net.load_state_dict(checkpoint['gen_state_dict'])
gen_net.eval()

trial_number=304
def gen_ttscgan_(c):
    fake_noise = torch.from_numpy(np.random.normal(0, 1, (trial_number, args.latent_dim))).type(Tensor).to('cpu')
    f_label_cls = (torch.ones((trial_number,)) * c).type(LongTensor).to('cpu')
    fake_sigs = gen_net(fake_noise, f_label_cls)
    tmp=fake_sigs.detach().numpy().squeeze()
    return tmp # (304, 10, 500)

for c in range(5):
    tmp=gen_ttscgan_(c)
    np.save(save_gen_dir + 'gen_class_'+str(c)+'_' + str(epoch + 1) + '.npy', tmp)

plot_class0.append(gen_ttscgan_(0))
np.save(tmp_dir + 'bad_example.npy', tmp)
# Item: delete below #
from sklearn.preprocessing import MinMaxScaler
ax.clear()
a=tmp[0,0,:200]
scaler = MinMaxScaler(feature_range=(0,1))
dataa = scaler.fit_transform((a.reshape(-1,1)))
dd=dataa.squeeze()/2+0.2
ax.plot(dd)

a=tmp[0,1,:200]
scaler = MinMaxScaler(feature_range=(0,1))
dataa = scaler.fit_transform((a.reshape(-1,1)))
dd=dataa.squeeze()/2
ax.plot(dd)

fig.savefig(result_dir+'DA/plots/bad_example.pdf')
# Item: delete above

## filter the generated noisy data
import mne
ch_names=['ch_'+str(i) for i in range(208)]
sfreq=2000
info=mne.create_info(ch_names, sfreq, ch_types='misc')
epochs=mne.EpochsArray(gen_data,info)
gen_data_filt=epochs.filter(l_freq=None,h_freq=300,picks='all')

fig,ax=plt.subplots()
itrial=0
for i in range(10):
    ax.plot(gen_data_filt.get_data()[itrial,i,:])


#### tSNE visualization: original, cwgangp, NI, ttscgan ####

from gesture.DA.metrics.visualization import visualization
labels=['orig','cwgangp','NI','ttscgan']
fig=visualization([plot_class0[i].transpose(0,2,1) for i in range(len(plot_class0))],'tSNE',labels)
save_dir=result_dir + 'DA/plots/'
filename = save_dir + 'tSNE_orig_NI.pdf'
fig.savefig(filename)

filename = save_dir + 'tSNE_TTSCGAN.pdf'
fig.savefig(filename)

#####  visualization the raw time series  ####
fig,ax=plt.subplots()
ax.clear()
offset=[-4,0,4]

# original
for i in range(3):
    ax.plot(plot_class0[0][0,i,:]+offset[i])
save_dir=result_dir + 'DA/plots/'
filename = save_dir + 'timeseries_original.pdf'
fig.savefig(filename)
# NI
# original
for i in range(3):
    ax.plot(plot_class0[1][0,i,:]+offset[i])
save_dir=result_dir + 'DA/plots/'
filename = save_dir + 'timeseries_NI.pdf'
fig.savefig(filename)

# CWGANGP
# original
for i in range(3):
    ax.plot(gen_data[0,i,:]+offset[i])
save_dir=result_dir + 'DA/plots/'
filename = save_dir + 'timeseries_CWGANGP.pdf'
fig.savefig(filename)

# TTSCGAN
for i in range(3):
    ax.plot(plot_class0[0][0,i+5,:]+offset[i])
save_dir=result_dir + 'DA/plots/'
filename = save_dir + 'timeseries_VAE.pdf'
fig.savefig(filename)
