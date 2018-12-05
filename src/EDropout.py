import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat

# modifed from https://github.com/tylergenter/pytorch/blob/42459c2544bc6c2e37c3459caaeca7c9eb1a8906/torch/nn/_functions/dropout.py
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EDropout(InplaceFunction):

    def __init__(self, p=0.5, train=False, inplace=False):
        super(EDropout, self).__init__()
        self.train = train
        self.inplace = inplace
        self.mask = None
        self.retain_p = None
        self.entropy_list = []
        self.peek_peb = 0.0

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
                assert self.retain_p is not None
                self.noise = self.mask

                offset = torch.zeros(self.retain_p.size())
                for i1, i2 in (self.retain_p == 0).nonzero():
                    offset[i1, i2] += 1e-4
                offset = offset.to(device)
                self.retain_p.add_(offset)
                self.noise.div_(self.retain_p)
            output.mul_(self.noise)

        return output

    def backward(self, grad_output):
        if self.train:
            return grad_output.mul(self.noise)
        else:
            return grad_output
