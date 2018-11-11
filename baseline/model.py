import torch.nn as nn
import torch


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


CNN_2_model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=96,
              kernel_size=5, padding=1, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=96, out_channels=128,
              kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=128, out_channels=256,
              kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    Flatten(),
    nn.Linear(in_features=1024, out_features=2048),
    nn.ReLU(),
    nn.Linear(in_features=2048, out_features=2048),
    nn.ReLU(),
    nn.Linear(in_features=2048, out_features=10),
)

CNN_2_model_standard_dropout = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=96,
              kernel_size=5, padding=1, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=96, out_channels=128,
              kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=128, out_channels=256,
              kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    Flatten(),
    nn.Linear(in_features=1024, out_features=2048),
    nn.ReLU(),

    nn.Dropout(0.5),

    nn.Linear(in_features=2048, out_features=2048),
    nn.ReLU(),
    nn.Linear(in_features=2048, out_features=10),
)

CNN_2_model_standard_dropout_before = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=96,
              kernel_size=5, padding=1, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=96, out_channels=128,
              kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=128, out_channels=256,
              kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    Flatten(),

    nn.Dropout(p=0.5),

    nn.Linear(in_features=1024, out_features=2048),
    nn.ReLU(),

    nn.Linear(in_features=2048, out_features=2048),
    nn.ReLU(),
    nn.Linear(in_features=2048, out_features=10),
)

CNN_2_model_2fcs = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=96,
              kernel_size=5, padding=1, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=96, out_channels=128,
              kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=128, out_channels=256,
              kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    Flatten(),
    nn.Linear(in_features=1024, out_features=2048),
    nn.ReLU(),
    nn.Linear(in_features=2048, out_features=10),
)
