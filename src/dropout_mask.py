import torch
from torch.distributions.bernoulli import Bernoulli

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DropoutMask():
    @staticmethod
    def mask(pebs, P=0.5):
        # input is a tensor with shape (batch_size, num neuron in layer)
        retain_p = DropoutMask.calc_p_matrix(pebs, P)
        # calc mask with bernoulli distribution
        mask = Bernoulli(retain_p).sample()
        return mask

    @staticmethod
    def calc_p_matrix(pebs, P):
        batch_size, num_neuron = pebs.size(0), pebs.size(1)
        ones = torch.ones(batch_size, num_neuron).to(device)

        upper = torch.mul(ones * (1 - P) * (num_neuron - 1), pebs)
        lower = torch.mul(((ones * (1 - P) * num_neuron) - 1), pebs) + P
        result = 1 - torch.div(upper, lower)
        return result