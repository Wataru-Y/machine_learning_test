import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Simple_Net(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Simple_Net, self).__init__()
        
        self.fc1 = nn.Linear(in_feature, 64)
        self.fc2 = nn.Linaer(64,64)
        self.fc3 = nn.Linaer(64,64)
        self.fc4 = nn.Linaer(64,out_feature)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x
