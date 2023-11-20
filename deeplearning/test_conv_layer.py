from common_dl import *

# aSection: implement one single filter on multiple channel data using shared weight method ####
shape=(32,10,1,500)
real=torch.ones(shape)
input=torch.randn(shape)
#input=input.squeeze()

class aaa(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Conv2d(10,10,(1,10),padding='same',padding_mode='reflect',bias=False,groups=10)
    def forward(self,x):
        for i in range(self.l1.weight.shape[0]):
            self.l1.weight.data[i,:,:,:]=self.l1.weight.data[0,:,:,:]
        output=self.l1(input)
        return output

net=aaa()
crit=nn.MSELoss()
opt=torch.optim.Adam(net.parameters())
epochs=10
for epoch in range(epochs):
    print('looping...')
    x=input
    net.train()
    opt.zero_grad()
    output=net(x)
    err=crit(real,output)
    err.backward()
    opt.step()




