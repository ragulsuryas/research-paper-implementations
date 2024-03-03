'''
Pytorch implementation of VGG architecture
'''

#Imports
import torch
import torch.nn as nn
import torchvision.models as models

architectures = {  'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] }

# fully connected layers 4096*4096*1000

class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(architecture=architectures['VGG19'])

        self.fcs = nn.Sequential(nn.Linear((512*7*7), 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, num_classes))
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        return x
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                relu = nn.ReLU()
                layers += [conv, relu]
                in_channels = out_channels
            else:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                layers += [max_pool]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG(in_channels=3, num_classes=10).to(device=device)
    batch_size = 16
    x = torch.rand(batch_size, 3, 224, 224).to(device=device)
    y = model(x)
    print(f'Output shape: {y.shape}')
    # Load pre-trained VGG model
    # pretrained_vgg = models.vgg16(pretrained=True)