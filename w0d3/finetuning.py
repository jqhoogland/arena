#%%
import time
import os
import copy

import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from arena.w0d3.resnet import my_resnet

# %%


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "../w0d3/data/hymenoptera_data"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# %%

