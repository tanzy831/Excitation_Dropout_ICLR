import torch
import torch.nn
import numpy as np
from inferno.io.box.cifar10 import get_cifar10_loaders
from model import *
from tqdm import tqdm
import timeit
import time
from pathlib import Path

DATASET_DIRECTORY = 'data'
MODEL_SAVE_DIRECTORY = 'models'
DOWNLOAD_CIFAR = True
EPOCH = 100
BATCH_SIZE = 100
VALID_BATCH_SIZE = 100

model_save_path = Path('./' + MODEL_SAVE_DIRECTORY)
if not (model_save_path.exists() and model_save_path.is_dir()):
    model_save_path.mkdir()

# create directory for this run
run_directory_path = model_save_path / str(time.time()).replace('.', '')
run_directory_path.mkdir()

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

    # save model
    torch.save(model.state_dict(), str(run_directory_path) +
               '/' + 'epoch_' + str(e + 1) + '.pt')
    log_str = 'Epoch {e}, Epoch training time: {train_time}, loss per sample: {LPS}, Accuracy: {acc}'.format(
        e=(e + 1),
        train_time=(e_end - e_start),
        LPS=(epoch_loss / (batch_idx * BATCH_SIZE)),
        acc=(correct / (len(validate_loader) * VALID_BATCH_SIZE)))

    with open(str(run_directory_path) + '/train_log.txt', 'a') as f:
        f.write(log_str + '\n')
    print(log_str)