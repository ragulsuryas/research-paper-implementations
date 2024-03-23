import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=6, 
                               kernel_size=(5, 5), 
                               stride=(1, 1), 
                               padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, 
                               out_channels=16, 
                               kernel_size=(5, 5), 
                               stride=(1, 1), 
                               padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, 
                               out_channels=120, 
                               kernel_size=(5, 5), 
                               stride=(1, 1), 
                               padding=(0, 0))
        self.fc1 = nn.Linear(in_features=(120*1*1), out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__": 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNet().to(device)
    x = torch.rand((32, 1, 32, 32)).to(device)
    y = model(x)
    print(y.shape)