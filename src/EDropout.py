import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat

# modifed from https://github.com/tylergenter/pytorch/blob/42459c2544bc6c2e37c3459caaeca7c9eb1a8906/torch/nn/_functions/dropout.py

class EDropout(InplaceFunction):

    def __init__(self, p=0.5, train=False, inplace=False):
        super(EDropout, self).__init__()
        self.train = train
        self.inplace = inplace
        self.mask = None

    def _make_noise(self, input):
        return input.new().resize_as_(input)

    def forward(self, input):
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if self.train:
            self.noise = torch.ones(input.size())
            if self.mask is not None:
                self.noise = self.mask
            output.mul_(self.noise)

        return output

    def backward(self, grad_output):
        if self.train:
            return grad_output.mul(self.noise)
        else:
            return grad_output
