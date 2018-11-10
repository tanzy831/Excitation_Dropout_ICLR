import torch.nn as nn
from inferno.io.box.cifar import get_cifar10_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.reshape import Flatten

# Fill these in:
LOG_DIRECTORY = 'logs'
SAVE_DIRECTORY = 'models'
DATASET_DIRECTORY = 'data'
DOWNLOAD_CIFAR = True
USE_CUDA = True

# Build torch model
model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, padding=1, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, padding=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, padding=0, stride=2),

    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=1),
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
  .build_optimizer('Adam') \
  .validate_every((2, 'epochs')) \
  .save_every((5, 'epochs')) \
  .save_to_directory(SAVE_DIRECTORY) \
  .set_max_num_epochs(10) \
  .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                  log_images_every='never'),
                log_directory=LOG_DIRECTORY)

# Bind loaders
trainer \
    .bind_loader('train', train_loader) \
    .bind_loader('validate', validate_loader)

if USE_CUDA:
  trainer.cuda()

# Go!
trainer.fit()
