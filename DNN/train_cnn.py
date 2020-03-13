import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from dnn_model import *

BATCH = 1024
DECAY = 0.005
LR = 0.0001
Epoch = 100

PATH = "../dataset/CNN"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root=PATH, train=True, download=True, transform = transform)
trainloader = DataLoader(train_data, batch_size = BATCH, shuffle = True, num_workers = 2)

test_data = datasets.MNIST(root=PATH, train=False, download=True, transform = transform)
testloader = DataLoader(test_data, batch_size = BATCH, shuffle = False, num_workers = 2)

device = torch.device("cuda")
net = Simple_CNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

for epoch in range(Epoch):

    net.train()

    with tqdm(trainloader, ncols=100) as pbar_train:

        for batch_i, (image, label) in enumerate(pbar_train):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(image)
            _, preds = torch.max(output, 1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            pbar_train.set_postfix(OrderedDict(epoch="{:>3}".format(epoch),loss="{:.4f}".format(loss.item())))

    with torch.no_grad():
        net.eval()
        val_loss = 0
        val_acc = 0
        count = 0
        for batch_i, (image, label) in enumerate(testloader):
            image, label = image.to(device), label.to(device)
            output = net(image)
            _, preds = torch.max(output, 1)
            loss = criterion(output, label)
            val_loss += loss.item()*image.size(0)
            val_acc += (torch.sum(preds==label.data)).item()
            count += image.size(0)

        print('eval_loss: %.3f' % (val_loss / count))
        print(' eval_acc: %.3f' % ((val_acc / count) * 100))
