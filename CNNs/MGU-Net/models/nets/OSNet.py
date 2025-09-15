import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nets.MGUNet import MGUNet_2

class OSMGUNet(nn.Module):
    def __init__(self, n_classes=10):  # n_classes dinamico
        super(OSMGUNet, self).__init__()
        self.stage = MGUNet_2(in_channels=1, n_classes=n_classes, feature_scale=4)
    
    def forward(self, inputs):
        out = self.stage(inputs)
        output = F.log_softmax(out, dim=1)
        return output