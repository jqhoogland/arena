#%%
import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import PIL
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import torchvision
from PIL import Image
from plotly.subplots import make_subplots
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from arena.w0d3 import utils

# %%

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU() 
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(3136, 128)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

# %%

model = ConvNet()
# %%

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset =  datasets.MNIST(root="./data", train=False, transform=transform, download=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

# %%

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

# %%

epochs = 3
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

MODEL_FILENAME = "./w1d2_convnet_mnist.pt"
device = "cuda" if t.cuda.is_available() else "cpu"

def train_convnet(trainloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    '''

    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []
    accuracy_list = []

    for epoch in tqdm(range(epochs)):

        for (x, y) in tqdm(trainloader, leave=False):

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss: t.Tensor = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())

        accuracy = 0

        for (x, y) in tqdm(testloader, leave=False):
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            accuracy += (y_hat.argmax(dim=1) == y).sum().item()

        accuracy /= len(testloader.dataset)
        accuracy_list.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy:.6f}")
        # TODO: print accuracy!

    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)    
    return loss_list

# %%

loss_list, accuracy_list = train_convnet(trainloader, epochs, loss_fn)

# %%

utils.plot_loss_and_accuracy(loss_list, accuracy_list)

# fig = px.line(y=loss_list, template="simple_white")
# fig.update_layout(title="Cross entropy loss on MNIST", yaxis_range=[0, max(loss_list)])
# fig.show()
# %%

class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x
# %%

class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features

        self.num_batches_tracked = t.tensor(0, dtype=t.float)

        super().__init__()

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        
        if not self.training:
            x_hat = (x - self.running_mean) / t.sqrt(self.running_var * self.eps)
        else:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            x_hat = (x - batch_mean) / t.sqrt(batch_var * self.eps)
            self.num_batches_tracked += 1

        print(x_hat.shape, self.weight.shape, self.bias.shape)

        return x_hat * self.weight.as_strided(size=x_hat.shape, stride=(0, 1, 0, 0)) \
            + self.bias.as_strided(size=x_hat.shape, stride=(0, 1, 0, 0))

    def extra_repr(self) -> str:
        return f"momentum={self.momentum}, eps={self.eps}, num_features={self.num_features}"

utils.test_batchnorm2d_module(BatchNorm2d)
utils.test_batchnorm2d_forward(BatchNorm2d)
utils.test_batchnorm2d_running_mean(BatchNorm2d)
# %%
