from model import *
import torch

model = CNN_2_model
input = torch.ones(1, 3, 32, 32)
output = model(input)
print(output.shape)
