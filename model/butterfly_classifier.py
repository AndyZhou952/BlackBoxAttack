import torch
import torch.nn.functional as F
from torch import nn

# Convolutional block with layer norm
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, padding=0, stride=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, x):
        x = self.conv(x)
        # x = self.norm(x)
        x = self.dropout(x)
        return x

# CNN Classifier
class Classifier(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=3, dropout=0, padding=1, stride=1, bias=True)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, dropout=0, padding=1, stride=1, bias=True)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, dropout=0, padding=1, stride=1, bias=True)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, dropout=0, padding=1, stride=1, bias=True)
        self.fc = torch.nn.LazyLinear(out_features=num_classes)
        
    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.max_pool2d(x ,kernel_size=2)
        
        x = F.silu(self.conv2(x))
        x = F.max_pool2d(x ,kernel_size=2)
        
        x = F.silu(self.conv3(x))
        x = F.max_pool2d(x ,kernel_size=2)
        
        x = F.silu(self.conv4(x))
        x = F.max_pool2d(x ,kernel_size=2)
        
        x = x.view(x.size()[0], -1)
        logits = self.fc(x)
        return logits

# Densenet121
class DenseNet121(nn.Module):

    def __init__(self, num_classes, model='densenet121', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model, pretrained, in_chans=3)
        self.fc1 = nn.Linear(1000,32)
        self.fc2 = nn.Linear(32,64)        
        self.fc3 = nn.Linear(64,num_classes)

        
    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x