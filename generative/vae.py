# %%
import os

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from einops.layers.torch import Rearrange
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm.notebook import tqdm

# %%


class LinearAutoencoder(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        super().__init__()

        encoder = []
        decoder = []

        delta_size = (input_size - hidden_size) // num_layers

        for i in range(num_layers - 1):
            encoder.append(
                nn.Linear(
                    input_size - delta_size * i, input_size - delta_size * (i + 1)
                )
            )
            encoder.append(nn.ReLU())

            decoder.append(
                nn.Linear(
                    input_size - delta_size * (i + 1), input_size - delta_size * i
                )
            )
            decoder.append(nn.ReLU())

        encoder.append(
            nn.Linear(input_size - delta_size * (num_layers - 1), hidden_size)
        )
        decoder.append(
            nn.Linear(hidden_size, input_size - delta_size * (num_layers - 1))
        )

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*reversed(decoder))

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.view(x.size(0), -1)

        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)

        x_reconstructed = x_reconstructed.reshape((-1, 28, 28))
        return x_reconstructed


# %%

# Load MNIST dataset

from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

transform = Compose(
    [
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
    ]
)

train_data = datasets.MNIST(
    root=os.path.dirname(__file__),
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.MNIST(
    root=os.path.dirname(__file__),
    train=False,
    download=True,
    transform=transform,
)

# %%

# Create data loaders.

batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# %%

# Get cpu or gpu device for training.
device = "cuda" if t.cuda.is_available() else "cpu"

# Train the model
linear_model = LinearAutoencoder(784, 64, 3).to(device)

# %%


def get_unique_test_images():
    test_images = []
    for i in range(10):
        for data, target in test_dataloader:
            if target[0] == i:
                test_images.append(data[0])
                break

    return t.stack(test_images)


def test_model_reconstruction(model: nn.Module, test_images: t.Tensor | None = None):
    test_images = test_images or get_unique_test_images()

    with t.no_grad():
        reconstructed = model(test_images).cpu()

        fig, axes = plt.subplots(
            nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4)
        )

        for images, row in zip([test_images, reconstructed], axes):
            for img, ax in zip(images, row):
                ax.imshow(img.view((28, 28)).numpy(), cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.show()


def train_model(model: nn.Module, num_epochs: int = 10):
    opt = t.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = F.mse_loss
    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data = data.to(device)

            # Forward pass
            output = model(data)
            loss = loss_fn(output, data)

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                )

        test_model_reconstruction(conv_model)


# %%


class Debug(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        print(x.shape)
        return x


class ConvAutoencoder(nn.Module):
    input_size: int
    hidden_size: int
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 0),
            # Debug(),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 2),
            # Debug(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 7 * 7),
            nn.ReLU(),
            Rearrange("b (ic h w) -> b ic h w", h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)
        return x_reconstructed


# %%

conv_model = ConvAutoencoder(784, 10).to(device)

train_model(conv_model)

# %%


def generate_images(model: LinearAutoencoder | ConvAutoencoder):
    # Choose number of interpolation points
    n_points = 11
    latent_dim_size = model.hidden_size

    # Constructing latent dim data by making two of the dimensions vary independently between 0 and 1
    latent_dim_data = t.zeros((n_points, n_points, latent_dim_size), device=device)
    x = t.linspace(-1, 1, n_points)
    latent_dim_data[:, :, 0] = x.unsqueeze(0)
    latent_dim_data[:, :, 1] = x.unsqueeze(1)
    # Rearranging so we have a single batch dimension
    latent_dim_data = einops.rearrange(
        latent_dim_data, "b1 b2 latent_dim -> (b1 b2) latent_dim"
    )
    print(latent_dim_data.shape)
    # Getting model output, and normalising & truncating it in the   range [0, 1]
    output = model.decoder(latent_dim_data).detach().cpu().numpy()
    output_truncated = np.clip((output * 0.3081) + 0.1307, 0, 1)

    print(output_truncated.shape)
    output_single_image = einops.rearrange(
        output_truncated,
        "(b1 b2) 1 height width -> (b1 height) (b2 width)",
        b1=n_points,
        b2=n_points,
    )

    # Plotting results
    fig = plt.matshow(output_single_image)
    plt.title("Decoder output from varying first two latent space dims")
    plt.show()


# Compress, then use PCA to reduce to 2 dimensions


def compress_and_reduce(model: LinearAutoencoder | ConvAutoencoder):
    # load all test images
    test_images, test_labels = (
        test_data.data.reshape((-1, 1, 28, 28)).float(),
        test_data.targets,
    )

    with t.no_grad():
        compressed = model.encoder(test_images.to(device)).cpu()

        pca = PCA(n_components=2)
        compressed_reduced = pca.fit_transform(compressed)

        fig, ax = plt.subplots()
        ax.scatter(compressed_reduced[:, 0], compressed_reduced[:, 1], c=test_labels)
        plt.show()


# %%

generate_images(conv_model)
compress_and_reduce(conv_model)

# %%


class VariationalEncoder(nn.Module):
    pre_encoder: nn.Sequential
    W_mu: nn.Linear
    W_logvar: nn.Linear

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * hidden_size),
        )


    def full_forward(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        std = t.exp(0.5 * logvar)
        eps = t.randn_like(std)

        return mu + eps * std, mu, logvar

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.full_forward(x)[0]


class VariationalAutoencoder(nn.Module):
    input_size: int
    hidden_size: int
    decoder: nn.Sequential

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        super().__init__()

        self.encoder = VariationalEncoder(input_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 7 * 7),
            nn.ReLU(),
            Rearrange("b (ic h w) -> b ic h w", h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
        )

    def full_forward(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        x_compressed, mu, logvar = self.encoder.full_forward(x)
        x_reconstructed = self.decoder(x_compressed)

        return x_reconstructed, mu, logvar

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.full_forward(x)[0]


# %%


def kl_loss(mu: t.Tensor, logvar: t.Tensor) -> t.Tensor:
    return 0.5 * t.sum(mu.pow(2) + logvar.exp() - logvar - 1)


def vae_loss(
    x_hat: t.Tensor,
    x: t.Tensor,
    mu: t.Tensor,
    logvar: t.Tensor,
    kl_div_coeff: float = 5.,
) -> t.Tensor:
    reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum")
    kl_divergence = kl_loss(mu, logvar)

    return reconstruction_loss + kl_div_coeff * kl_divergence


def train_vae(
    model: VariationalAutoencoder,
    num_epochs: int = 5,
    opt: t.optim.Optimizer | None = None,
):
    opt = opt or t.optim.Adam(model.parameters(), lr=1e-4)

    test_model_reconstruction(model)

    for epoch in tqdm(range(num_epochs), desc="Training..."):
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data = data.to(device)

            # Forward pass
            reconstruction, mu, logvar = model.full_forward(data)
            loss = vae_loss(reconstruction, data, mu, logvar)

            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f} (kl: {kl_loss(mu, logvar).item():.4f})"
                )

        test_model_reconstruction(model)


# %%

vae_model = VariationalAutoencoder(784, 5).to(device)
train_vae(vae_model, 5)

# %%


generate_images(vae_model)
compress_and_reduce(vae_model)

# %%
