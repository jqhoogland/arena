#%%
import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import PIL
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import torchvision
from arena.w0d3 import utils
from PIL import Image
from plotly.subplots import make_subplots
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# %%

class AlexNet(nn.Module):
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

model = AlexNet()
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

epochs = 3
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

MODEL_FILENAME = "./w1d2_AlexNet_mnist.pt"
device = "cuda" if t.cuda.is_available() else "cpu"

def train_AlexNet(trainloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    '''
    Defines a AlexNet using our previous code, and trains it on the data in trainloader.
    '''

    model = AlexNet().to(device).train()
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

loss_list, accuracy_list = train_AlexNet(trainloader, epochs, loss_fn)

# %%

utils.plot_loss_and_accuracy(loss_list, accuracy_list)

# fig = px.line(y=loss_list, template="simple_white")
# fig.update_layout(title="Cross entropy loss on MNIST", yaxis_range=[0, max(loss_list)])
# fig.show()
# %%
