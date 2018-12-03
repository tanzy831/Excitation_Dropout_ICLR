import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Flatten
from EDropout import *
import excitationbp as eb
import copy
from torch.autograd import Variable
from dropout_mask import DropoutMask

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class latterModel(nn.Module):
    def __init__(self, f1, f2):
        super(latterModel, self).__init__()
        self.fc1 = f1
        self.fc2 = f2
    
    def forward(self, x):
        h6 = F.relu(self.fc1(x))
        y = self.fc2(h6)
        return y


class CNN_2_EDropout(nn.Module):
    def __init__(self, batch_size):
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

        self.ed = EDropout(p=0.5, train=self.training, inplace=True)

        self.batch_size = batch_size


    def forward(self, x, label):
        h1 = self.maxpool1(F.relu(self.cnn1(x)))
        h2 = self.maxpool2(F.relu(self.cnn2(h1)))
        h3 = self.maxpool3(F.relu(self.cnn3(h2)))
        h4 = self.flatten(h3)
        h5 = F.relu(self.fc1(h4))

        model2 = latterModel(copy.deepcopy(self.fc2), copy.deepcopy(self.fc3))
        eb.use_eb(True, verbose=False)
        peb_list = []
        mask = None
        retain_p = None
        if self.training:
            data = h5.clone()
            for i in range(self.batch_size):
                prob_outputs = Variable(torch.zeros(1, 10)).to(device)
                prob_outputs.data[:, label[i]] += 1

                prob_inputs = eb.excitation_backprop(
                    model2, data[i:i + 1, :], prob_outputs, 
                    contrastive=False, target_layer=0)
                peb_list.append(prob_inputs)
        
            pebs = torch.cat(peb_list, dim=0) # calc peb
            mask, retain_p = DropoutMask.mask(pebs) # calc mask
            eb.use_eb(False, verbose=False)

        self.ed.train = self.training  # ugly code!
        self.ed.mask =  mask
        self.ed.retain_p = retain_p
        h_ed = self.ed(h5)

        h6 = F.relu(self.fc2(h_ed))
        h7 = self.fc3(h6)

        return h7
