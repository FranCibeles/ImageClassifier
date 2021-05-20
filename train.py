import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import utils

ap = argparse.ArgumentParser(description='Train.py')

ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=2)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/", type = str)

pa = ap.parse_args()
where = ''.join(pa.data_dir)
path = pa.save_dir
lr = pa.learning_rate
epochs = pa.epochs

trainloader, validloader, testloader, train_data, valid_data, test_data = utils.load_data(where)

model, optimizer, criterion = utils.nn_setup(lr)

utils.checkpoint(train_data, model, optimizer, epochs)

print("You have trained your model")