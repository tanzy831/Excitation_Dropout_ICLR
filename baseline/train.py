import torch.nn as nn
from inferno.io.box.cifar10 import get_cifar10_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.reshape import Flatten
import torch
import numpy as np
from inferno.trainers.callbacks.base import Callback

# Fill these in:
LOG_DIRECTORY = 'logs'
SAVE_DIRECTORY = 'models'
DATASET_DIRECTORY = 'data'
DOWNLOAD_CIFAR = True
USE_CUDA = torch.cuda.is_available()


class LossLogger(Callback):
    def end_of_training_iteration(self, **_):
        # The callback object has the trainer as an attribute.
        # The trainer populates its 'states' with torch tensors (NOT VARIABLES!)
        training_loss = self.trainer.get_state('training_loss')
        # Extract float from torch tensor
        training_loss = training_loss[0]
        print(training_loss)


# Build torch model
model = nn.Sequential(
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
    nn.Softmax(),
)

# Load loaders
train_loader, validate_loader = get_cifar10_loaders(DATASET_DIRECTORY,
                                                    download=DOWNLOAD_CIFAR)

# Build trainer
trainer = Trainer(model) \
  .build_criterion('CrossEntropyLoss') \
  .build_metric('CategoricalError') \
  .evaluate_metric_every((10, 'iterations')) \
  .build_optimizer('Adam') \
  .validate_every((2, 'epochs')) \
  .save_every((5, 'epochs')) \
  .save_to_directory(SAVE_DIRECTORY) \
  .set_max_num_epochs(10) \
  .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                  log_images_every='never'),
                log_directory=LOG_DIRECTORY)

trainer.register_callback(LossLogger())

# Bind loaders
trainer \
    .bind_loader('train', train_loader) \
    .bind_loader('validate', validate_loader)

if USE_CUDA:
    trainer.cuda()

# Go!
trainer.fit()
