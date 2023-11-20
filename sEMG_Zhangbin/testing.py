import torch
from gesture.models.d2l_resnet import d2lresnet
import numpy as np
from common_dl import myDataset, set_random_seeds
from torch.utils.data import DataLoader
from tqdm import tqdm

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

result_path = 'H:/Long/data/sEMG_Zhangbin/training/'
data_dir = 'H:/Long/data/sEMG_Zhangbin/'
model_name = 'resnet' #'deepnet'
net=d2lresnet(as_DA_discriminator=False) # 92%
lr = 0.01

savepath = result_path + 'checkpoint_resnet_3.pth'
#torch.save(state, savepath)

checkpoint = torch.load(savepath)
net.load_state_dict(checkpoint['net'])
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
optimizer.load_state_dict(checkpoint['optimizer'])


[ET_test,PD_test,others_test,NC_test] = np.load(data_dir+'dataset_test.npy', allow_pickle=True)
others_test_y=[2]*len(others_test)
PD_test_y=[1]*len(PD_test)
ET_test_y=[0]*len(ET_test)
NC_test_y=[3]*len(NC_test)

batch_size=320
X_test=np.concatenate((ET_test,PD_test,others_test,NC_test))
y_test=ET_test_y+PD_test_y+others_test_y+NC_test_y
test_set=myDataset(X_test,np.asarray(y_test)) #len(test_set):72606
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_size=len(test_loader.dataset) #190

net.eval()
y_hat=[]
y_target=[]
# print("Validating...")
with torch.no_grad():
    running_corrects = 0
    for _, (test_x, test_y) in enumerate(tqdm(test_loader)):
        y_target.append(test_y.tolist())
        if (cuda):
            test_x = test_x.float().cuda()
            # val_y = val_y.float().cuda()
        else:
            test_x = test_x.float()
            # val_y = val_y.float()
        outputs = net(test_x)
        #_, preds = torch.max(outputs, 1)
        preds = outputs.argmax(dim=1, keepdim=True)
        y_hat.append([elei for listi in preds.cpu().tolist() for elei in listi])

        running_corrects += torch.sum(preds.cpu().squeeze() == test_y.squeeze())
test_acc = (running_corrects.double() / test_size).item()
print("Test accuracy: {:.2f}.".format(test_acc))

