import scipy.stats

class Entropy_Calc():

    @staticmethod
    def peb_entropy_calc(pebs):
        entropy=scipy.stats.entropy(pebs)  # input probabilities to get the entropy 
        return entropy

    @staticmethod
    def activation_entropy_calc(activations):
        pass