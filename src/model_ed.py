import torch.nn as nn
import torch
import torch.nn.functional as F
# from ...Exictation_Dropout_ICLR.baseline.model import Flatten
from model import Flatten


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

    def forward(self, x):
        h1 = self.maxpool1(F.relu(self.cnn1(x)))
        h2 = self.maxpool2(F.relu(self.cnn2(h1)))
        h3 = self.maxpool3(F.relu(self.cnn3(h2)))

        h4 = self.flatten(h3)

        h5 = F.relu(self.fc1(h4))
        h6 = F.relu(self.fc2(h5))
        h7 = self.fc3(h6)
        return h7

    def forward_ed(self):
        h1 = self.maxpool1(F.relu(self.cnn1(x)))
        h2 = self.maxpool2(F.relu(self.cnn2(h1)))
        h3 = self.maxpool3(F.relu(self.cnn3(h2)))

        h4 = self.flatten(h3)
