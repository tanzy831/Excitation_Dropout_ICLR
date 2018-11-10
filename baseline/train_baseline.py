import torch
import torch.nn
import numpy as np
from inferno.io.box.cifar10 import get_cifar10_loaders
from model import *
from tqdm import tqdm
import timeit

DATASET_DIRECTORY = 'data'
DOWNLOAD_CIFAR = True
EPOCH = 100
BATCH_SIZE = 100
VALID_BATCH_SIZE = 100

train_loader, validate_loader = get_cifar10_loaders(
    DATASET_DIRECTORY, train_batch_size=BATCH_SIZE, test_batch_size=VALID_BATCH_SIZE, download=DOWNLOAD_CIFAR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN_2_model
model = model.to(device)

ADAMOptimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for e in range(EPOCH):
    epoch_loss = 0
    e_start = timeit.default_timer()
    batch_idx = 0
    for data, label in tqdm(train_loader, total=len(train_loader)):
        torch.cuda.empty_cache()
        data, label = data.to(device).float(), label.to(device).long()
        ADAMOptimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        l = loss.item()
        loss.backward()
        ADAMOptimizer.step()
        epoch_loss += l
        batch_idx += 1
    e_end = timeit.default_timer()

    # validation
    correct = 0
    for data, label in validate_loader:
        data, label = data.to(device).float(), label.to(device).long()
        output = model(data)
        result = output.max(1)[1]
        correct += label.eq(result).sum()
    correct = correct.float().cpu()
    print('Epoch training time: ', e_end - e_start, ', loss per sample: ', epoch_loss /
          (batch_idx * BATCH_SIZE), ', Accuracy: ', correct / (len(validate_loader) * VALID_BATCH_SIZE))
