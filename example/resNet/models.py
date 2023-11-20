import torch
from torch import nn
from torchvision.models.resnet import resnet18

'''
Here resNet can be defined with arbitrary input channels instead of only 3 RGB channel.
Also you can specify if load the pretrained model or not. If so, the extra channels should be initialized with some
pretrained channels.
'''
def my_resnet18(input_channels,num_classes,pretrained=False):
    if pretrained==False:
        model = resnet18(pretrained=False)
        # count_parameters(resnet18)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc=nn.Linear(in_features=512, out_features=num_classes, bias=True)
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        return model
    else:
        model = resnet18(pretrained=True)
        layer = model.conv1
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=input_channels,out_channels=layer.out_channels,kernel_size=layer.kernel_size,
                              stride=layer.stride,padding=layer.padding,bias=layer.bias)

        # initialized the extra channels with some or one pre-trained channel.
        copy_weights = 0

        # Copying the weights from the pretrained to the new layer
        new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        # Copying the weights of the `copy_weights` channel of the pretrained layer to the extra channels of the new layer
        for i in range(input_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, :,:].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        model.conv1 = new_layer
        model.fc=nn.Linear(in_features=512, out_features=num_classes, bias=True)
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        return model
# exam model
mm=my_resnet18(6,5, pretrained=True)

x= torch.randn(1, 6, 224, 224)
mm(x).shape



