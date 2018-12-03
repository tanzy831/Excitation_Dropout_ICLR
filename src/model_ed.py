import torch.nn as nn
import torch
import torch.nn.functional as F
# from ...Exictation_Dropout_ICLR.baseline.model import Flatten
from model import Flatten
import time


class CNN_2_EDropout(nn.Module):
    def __init__(self):
        super(CNN_2_EDropout, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=96,
                              kernel_size=5, padding=1, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        self.cnn2 = nn.Conv2d(in_channels=96, out_channels=128,
                              kernel_size=5, padding=2, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256,
                              kernel_size=5, padding=2, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)

        self.flatten = Flatten()

        self.fc1 = nn.Linear(in_features=1024, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=10)

        self.middle = None

    def forward(self, x, mask=None, non_eb=False):
        if mask is None:
            h1 = self.maxpool1(F.relu(self.cnn1(x)))
            h2 = self.maxpool2(F.relu(self.cnn2(h1)))
            h3 = self.maxpool3(F.relu(self.cnn3(h2)))

            h4 = self.flatten(h3)

            h5 = F.relu(self.fc1(h4))
            if non_eb:
                self.middle = h5
        else:
            assert self.middle.shape == mask.shape
            h5 = self.middle * mask
        h6 = F.relu(self.fc2(h5))
        h7 = self.fc3(h6)

        if mask is not None and non_eb:
            self.middle = None

        return h7

# class EDLinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super(EDLinear, self).__init__(in_features, out_features, bias)
    
#     def forward(self, input):
#         output = super(EDLinear, self).forward(input)
#         return output