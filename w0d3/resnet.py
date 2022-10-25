#%%
import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from pprint import pp

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

from torchvision import models

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


        super().__init__()

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0, dtype=t.float))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        
        if not self.training:
            x_hat = (x - self.running_mean.reshape((1, -1, 1, 1))) / t.sqrt(self.running_var + self.eps)
        else:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean[0, :, 0, 0]
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var[0, :, 0, 0]

            x_hat = (x - batch_mean) / t.sqrt(batch_var + self.eps)
            self.num_batches_tracked += 1

        return x_hat * self.weight.as_strided(size=x_hat.shape, stride=(0, 1, 0, 0)) \
            + self.bias.as_strided(size=x_hat.shape, stride=(0, 1, 0, 0))

    def extra_repr(self) -> str:
        return f"momentum={self.momentum}, eps={self.eps}, num_features={self.num_features}"

utils.test_batchnorm2d_module(BatchNorm2d)
utils.test_batchnorm2d_forward(BatchNorm2d)
utils.test_batchnorm2d_running_mean(BatchNorm2d)
# %%

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return x.mean(dim=(2, 3))


# %%


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        super().__init__()

        self.left = Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1, bias=False),
            BatchNorm2d(out_feats),
            nn.ReLU(),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_feats)
        )

        if first_stride > 1:
            self.right = nn.Sequential(
                nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, bias=False),
                BatchNorm2d(out_feats)
            )
        else:
            self.right = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        x = self.left(x) + self.right(x)
        x = self.relu(x)

        return x

print(pretrained_resnet)

#%%

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        self.n_blocks = n_blocks
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        super().__init__()

        self.blocks = nn.ModuleList([
            ResidualBlock(in_feats, out_feats, first_stride) if i == 0 else ResidualBlock(out_feats, out_feats)
            for i in range(n_blocks)
        ])

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        for block in self.blocks:
            x = block(x)

        return x


# %%

class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        super().__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        out_features_per_group = [out_features_per_group[0], *out_features_per_group]
        self.block_groups = nn.ModuleList([
            BlockGroup(n_blocks_per_group[i], out_features_per_group[i], out_features_per_group[i + 1], first_strides_per_group[i])
            for i in range(len(n_blocks_per_group))
        ])
        self.avgpool = AveragePool()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_features_per_group[-1], n_classes)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        '''
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for block_group in self.block_groups:
            x = block_group(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
# %%

my_resnet = ResNet34()
pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1, progress=False)

# %%

my_resnet_state = list(my_resnet.state_dict().items())
pretrained_resnet_state = list(pretrained_resnet.state_dict().items())

for i in range(min(len(my_resnet_state), len(pretrained_resnet_state))):

    if my_resnet_state[i][1].shape != pretrained_resnet_state[i][1].shape:
        print("ERROR:")
        print(my_resnet_state[i][0], my_resnet_state[i][1].shape)
        print(pretrained_resnet_state[i][0], pretrained_resnet_state[i][1].shape)

    else:
        print(my_resnet_state[i][0], pretrained_resnet_state[i][0], my_resnet_state[i][1].shape)

    print("\n")

# %%

def copy_weights(myresnet: ResNet34, pretrained_resnet: torchvision.models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    mydict = myresnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()

    # Check the number of params/buffers is correct
    assert len(mydict) == len(pretraineddict), "Number of layers is wrong. Have you done the prev step correctly?"

    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict_to_load = {}

    for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items()):
        state_dict_to_load[mykey] = pretrainedvalue

    myresnet.load_state_dict(state_dict_to_load)

    return myresnet

my_resnet = copy_weights(my_resnet, pretrained_resnet)
# %%

IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = Path("../w0d3/resnet_inputs")

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]
images[0]
# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# %%

def prepare_data(images: list[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    return t.stack([transform(image) for image in images])
    

prepared_images = prepare_data(images)
# %%

def predict(model, images):
    logits = model(images)
    return logits.argmax(dim=1)

# %%
IMAGENET_CLASSES = []

with open("../w0d3/imagenet_labels.json") as f:
    IMAGENET_CLASSES = list(json.load(f).values())

def predict_label(model, images):
    labels = predict(model, images)
    return [IMAGENET_CLASSES[label] for label in labels]

# %%

predict_label(my_resnet, prepared_images)
# %%

