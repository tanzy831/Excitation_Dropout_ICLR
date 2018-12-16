import scipy.stats
import torch
import numpy as np

class Entropy_Calc():

    @staticmethod
    def peb_entropy_calc(pebs):
        #pebs_list = pebs[0]
        # use 5 images combinations to calcualte average peb entropy 
        pebs_list = torch.cat((pebs[0], pebs[1], pebs[2], pebs[3], pebs[4]), dim=0)
        max_val = torch.max(pebs_list)
        min_val = torch.min(pebs_list)
        unit = (max_val - min_val) / 999
        occu_list = [0] * 1000
        for val in pebs_list:
            index = np.floor(((val - min_val) / unit).cpu().numpy()).astype(int)
            occu_list[index] += 1
        dist_list = [x / sum(occu_list) for x in occu_list]
        entropy=scipy.stats.entropy(occu_list)  # input probabilities to get the entropy 
        return entropy

    @staticmethod
    def activation_entropy_calc(activations):
        pass