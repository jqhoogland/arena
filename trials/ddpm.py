# %%

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import plotly.express as px
import torch as t
import torchinfo
import wandb
from torch import nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from arena.cv.diffusion.ddpm import DDPM
from arena.cv.diffusion.diffusion import (
    NoiseSchedule,
    log_images,
    noise_img,
    normalize_img,
    reconstruct,
)

device = "cuda" if t.cuda.is_available() else "cpu"
MAIN = __name__ == "__main__"
# %%


def get_fashion_mnist(
    train_transform, test_transform
) -> tuple[TensorDataset, TensorDataset]:
    """Return MNIST data using the provided Tensor class."""
    mnist_train = datasets.FashionMNIST("../cv/data", train=True, download=True)
    mnist_test = datasets.FashionMNIST("../cv/data", train=False)
    print("Preprocessing data...")
    train_tensors = TensorDataset(
        t.stack(
            [
                train_transform(img)
                for (img, label) in tqdm(iter(mnist_train), desc="Training data")
            ]
        )
    )
    test_tensors = TensorDataset(
        t.stack(
            [
                test_transform(img)
                for (img, label) in tqdm(iter(mnist_test), desc="Test data")
            ]
        )
    )
    return (train_tensors, test_tensors)


if MAIN:
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda t: t * 2 - 1),
        ]
    )
    data_folder = Path("./data/fashion_mnist")
    data_folder.mkdir(exist_ok=True, parents=True)
    DATASET_FILENAME = data_folder / "generative_models_dataset_fashion.pt"

    trainset, test_set = None, None

    if DATASET_FILENAME.exists():
        (trainset, testset) = t.load(str(DATASET_FILENAME))
    else:
        (trainset, testset) = get_fashion_mnist(train_transform, train_transform)
        t.save((trainset, testset), str(DATASET_FILENAME))

# %%


@dataclass
class DDPMConfig:
    model_channels: int = 28
    model_dim_mults: tuple[int, ...] = (1, 2, 4)
    image_shape: tuple[int, int, int] = (1, 28, 28)
    max_steps: int = 200
    epochs: int = 10
    lr: float = 0.001
    batch_size: int = 256
    img_log_interval: int = 200
    n_images_to_log: int = 3
    device: Literal["cpu", "cuda"] = "cpu"


def train(
    model: nn.Module,
    config: DDPMConfig,
    trainset: TensorDataset,
    testset: TensorDataset,
) -> None:
    wandb.init(project="diffusion_models", config=asdict(config))

    opt = t.optim.Adam(model.parameters(), lr=config.lr)
    train_loader = t.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = t.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=True
    )
    loss_fn = nn.MSELoss()

    schedule = model.noise_schedule

    n_examples_seen = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (img,) in enumerate(train_loader):
            img = img.to(config.device)
            num_steps, noise, noised = noise_img(
                normalize_img(img), schedule, max_steps=config.max_steps
            )
            noise_pred = model(noised, num_steps)
            loss = loss_fn(noise_pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            n_examples_seen += img.shape[0]

            wandb.log({"train_loss": loss.item()}, step=n_examples_seen)

            if (batch_idx + 1) % config.img_log_interval == 0:
                with t.inference_mode():
                    reconstructed = reconstruct(noised, noise, num_steps, schedule)
                    images = log_images(
                        img,
                        noised,
                        noise,
                        noise_pred,
                        reconstructed,
                        num_images=config.n_images_to_log,
                    )
                    wandb.log({"images": images}, step=n_examples_seen)

        train_loss /= len(train_loader)

        if testset:
            model.eval()
            test_loss = 0
            with t.inference_mode():
                for (img,) in test_loader:
                    img = img.to(config.device)
                    num_steps, noise, noised = noise_img(
                        normalize_img(img), schedule, max_steps=1000
                    )
                    noise_pred = model(noised, num_steps)
                    test_loss += loss_fn(noise_pred, noise)

                test_loss /= len(test_loader.dataset)

            wandb.log({"test_loss": test_loss}, step=n_examples_seen)

    wandb.save("./wandb/gradients.h5")
    wandb.finish()


if MAIN:
    config = DDPMConfig()
    model = DDPM(
        image_shape=config.image_shape,
        channels=config.model_channels,
        dim_mults=config.model_dim_mults,
        max_steps=config.max_steps,
    )
    model.noise_schedule = NoiseSchedule(
        config.max_steps,
        config.device,
    )

if MAIN:
    model = train(model, config, trainset, testset)
# %%


# TODO: Get this working. Too frustrating for now.
