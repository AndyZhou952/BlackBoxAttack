import torch
import torch.nn.functional as F
from torch import nn
import timm

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