import torch
from torch import nn

class tsception(nn.Module):
    def __init__(self, sampling_rate, chnNum,wind_size, num_T, num_S,dropout,mel_bins):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(tsception, self).__init__()
        # try to use shorter conv kernel to capture high frequency
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        win = [int(tt * sampling_rate) for tt in self.inception_window]
        # [500, 250, 125, 62, 31, 15, 7]
        # in order to have a same padding, all kernel size shoule be even number, then stride=kernel_size/2
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation

        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[0]), stride=1, padding=(0, 250)),  # kernel: 500
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[1]), stride=1, padding=(0, 125)),  # 250
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[2] + 1), stride=1, padding=(0, 63)),  # kernel: 126
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[3]), stride=1, padding=(0, 31)),  # kernel:62
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[4] + 1), stride=1, padding=(0, 16)),  # 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception6 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[5] + 1), stride=1, padding=(0, 8)),  # 15
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception7 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[6] + 1), stride=1, padding=(0, 4)),  # 7
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(chnNum, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(int(chnNum * 0.5), 1), stride=(int(chnNum * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception3 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(int(chnNum * 0.5 * 0.5), 1),
                      stride=(int(chnNum * 0.5 * 0.5), 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_S * 6)
        self.BN_s = nn.BatchNorm2d(num_S * 6)

        self.drop = nn.Dropout(dropout)
        self.avg = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))


        #### test the output shape ####
        x=torch.ones((1,1,chnNum,wind_size))
        if (len(x.shape) < 4):
            x=torch.unsqueeze(x,dim=1) # torch.Size([3, 1, 118, 200])
        y2 = self.Tception2(x) # torch.Size([28, 3, 198, 124])
        y3 = self.Tception3(x) # torch.Size([28, 3, 198, 124])
        y4 = self.Tception4(x) # same as previous
        y5 = self.Tception5(x) # ..
        y6 = self.Tception6(x) # ..
        y7 = self.Tception7(x) # .. # (batch_size, plan, channel, time)
        out = torch.cat((y2, y3, y4, y5, y6, y7), dim=1) # torch.Size([3, 36=num_T*6, 118, 24]) # concate alone plan:  torch.Size([28, 18, 198, 124])
        #out = self.BN_t(out) #Todo: braindecode didn't use normalization between t and s filter.

        z1 = self.Sception1(out) # torch.Size([3, 36, 1, 3])
        z2 = self.Sception2(out) # torch.Size([3, 36, 2, 3])
        z3 = self.Sception2(out) # torch.Size([3, 36, 2, 3])
        out_final = torch.cat((z1, z2, z3), dim=2) #torch.Size([3, 36, 5, 3])
        out = self.BN_s(out_final)

        out = out.permute(0, 3, 1, 2)  # (batchsize, seq, height, width), ([280, 38, 3, 7])
        seqlen = out.shape[1]
        input_size = int(out.shape[2] * out.shape[3])
        out = out.reshape(1, seqlen, input_size)  # torch.Size([3, 3, 180])

        h_in=out.shape[-1]
        ### test end ###

        self.lstm1 = nn.LSTM(h_in, 45, batch_first=True)

        self.linear1 = nn.Sequential(
            nn.Linear(45, mel_bins), # 80 mel bins
            #nn.ReLU()
        )



    def forward(self, x):  # ([128, 1, 4, 1024]): (batch_size, )
        #self.float()
        #x = torch.squeeze(x, dim=0)
        batch_size=x.shape[0]
        # y1 = self.Tception1(x)
        if (len(x.shape) < 4):
            x=torch.unsqueeze(x,dim=1) # torch.Size([3, 1, 118, 200])
        y2 = self.Tception2(x) # torch.Size([28, 3, 198, 124])
        y3 = self.Tception3(x) # torch.Size([28, 3, 198, 124])
        y4 = self.Tception4(x) # same as previous
        y5 = self.Tception5(x) # ..
        y6 = self.Tception6(x) # ..
        y7 = self.Tception7(x) # .. # (batch_size, plan, channel, time)
        out = torch.cat((y2, y3, y4, y5, y6, y7), dim=1) # torch.Size([3, 36=num_T*6, 118, 24]) # concate alone plan:  torch.Size([28, 18, 198, 124])
        #out = self.BN_t(out) #Todo: braindecode didn't use normalization between t and s filter.

        z1 = self.Sception1(out) # torch.Size([3, 36, 1, 3])
        z2 = self.Sception2(out) # torch.Size([3, 36, 2, 3])
        z3 = self.Sception2(out) # torch.Size([3, 36, 2, 3])
        out_final = torch.cat((z1, z2, z3), dim=2) #torch.Size([3, 36, 5, 3])
        out = self.BN_s(out_final)

        # Todo: test the effect of log(power)
        #out = torch.pow(out, 2) # ([28, 18, 5, 15])
        # Braindecode use avgpool2d here
        # out = self.avg(out)
        # out = torch.log(out)
        out=self.drop(out)


        out = out.permute(0, 3, 1, 2)  # (batchsize, seq, height, width), ([280, 38, 3, 7])
        seqlen = out.shape[1]
        input_size = int(out.shape[2] * out.shape[3])
        out = out.reshape(batch_size, seqlen, input_size)  # torch.Size([3, 3, 180])

        out, _ = self.lstm1(out)
        pred = self.linear1(torch.squeeze(out[:, -1, :]))
        pred = torch.squeeze(pred)
        #pred = torch.unsqueeze(pred, dim=0) # torch.Size([3, 80])
        return pred

class simple_fcnet(nn.Module):
    def __init__(self, input_d, output_d):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(simple_fcnet, self).__init__()
        self.input_d=input_d
        self.output_d=output_d
        self.fc1 = nn.Sequential(nn.Linear(self.input_d, 100), nn.ReLU(), nn.BatchNorm1d(100))
        self.fc2 = nn.Sequential(nn.Linear(100, self.input_d), nn.ReLu())
    def forward(self, x): # x:(batch, features)
        y=self.fc1(x)
        y=self.fc2(y)
        return y


