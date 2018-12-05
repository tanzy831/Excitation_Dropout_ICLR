import scipy.stats
import torch

class Entropy_Calc():

    @staticmethod
    def peb_entropy_calc(pebs):
        pebs_list = torch.reshape(pebs, (-1, ))
        entropy=scipy.stats.entropy(pebs_list)  # input probabilities to get the entropy 
        return entropy

    @staticmethod
    def activation_entropy_calc(activations):
        pass