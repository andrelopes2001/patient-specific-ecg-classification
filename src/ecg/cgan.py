"""Conditional GAN for minority-class beat synthesis (PyTorch).

Trained per class to synthesise extra S, V and F beats, addressing the 89%/0.8%
imbalance between N and F. Architecture as in the thesis: noise and one-hot
class context are embedded separately, concatenated, then decoded to a beat.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import CLASSES, DEFAULT, Config


def _block(in_features: int, out_features: int, activation, dropout: float = 0.2):
    return nn.Sequential(
        nn.Linear(in_features, out_features), nn.Dropout(dropout), activation
    )


class Generator(nn.Module):
    """Maps (noise, class) to a synthetic beat window."""

    def __init__(
        self,
        latent_dim: int = 151,
        context_dim: int = len(CLASSES),
        output_dim: int = 151,
        hidden: int = 1200,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.noise_embed = _block(latent_dim, 200, nn.ReLU())
        self.context_embed = _block(context_dim, 1000, nn.ReLU())
        self.hidden = _block(1200, hidden, nn.ReLU())
        # Sigmoid output: beats must be min-max scaled to [0, 1] to match.
        self.out = nn.Sequential(nn.Linear(hidden, output_dim), nn.Sigmoid())

    def forward(self, noise: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = torch.cat((self.noise_embed(noise), self.context_embed(context)), dim=1)
        return self.out(self.hidden(h))


class Discriminator(nn.Module):
    """Scores whether a (beat, class) pair is real."""

    def __init__(self, input_dim: int = 151, context_dim: int = len(CLASSES)):
        super().__init__()
        self.input_dim = input_dim

        self.beat_embed = _block(input_dim, 240, nn.LeakyReLU())
        self.context_embed = _block(context_dim, 50, nn.LeakyReLU())
        self.hidden = _block(290, 240, nn.LeakyReLU())
        self.out = nn.Sequential(nn.Linear(240, 1), nn.Sigmoid())

    def forward(self, beat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = torch.cat((self.beat_embed(beat), self.context_embed(context)), dim=1)
        return self.out(self.hidden(h))


def sample_noise(
    n: int, dim: int = 151, device: str | torch.device = "cpu"
) -> torch.Tensor:
    return torch.rand((n, dim), device=device)


def minibatch(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    n_classes: int = len(CLASSES),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random minibatch, with the labels one-hot encoded.

    Takes an explicit ``generator`` so batch selection is reproducible.
    """
    idx = torch.randperm(X.shape[0], generator=generator)[:batch_size]
    beats = torch.as_tensor(X[idx.numpy()], dtype=torch.float, device=device)
    beats = beats.reshape(len(idx), -1)
    labels = torch.as_tensor(y[idx.numpy()], dtype=torch.long, device=device)
    context = nn.functional.one_hot(labels, num_classes=n_classes).float()
    return beats, context


@torch.no_grad()
def generate(
    generator: Generator,
    labels: np.ndarray,
    device: str | torch.device = "cpu",
    n_classes: int = len(CLASSES),
) -> np.ndarray:
    """Synthesise one beat per entry of ``labels``."""
    generator.eval()
    context = nn.functional.one_hot(
        torch.as_tensor(labels, dtype=torch.long, device=device),
        num_classes=n_classes,
    ).float()
    noise = sample_noise(len(labels), generator.latent_dim, device)
    return generator(noise, context).cpu().numpy()


def build(cfg: Config = DEFAULT, device: str | torch.device = "cpu"):
    """Construct a generator/discriminator pair sized from config."""
    length = cfg.window_length
    return (
        Generator(latent_dim=length, output_dim=length).to(device),
        Discriminator(input_dim=length).to(device),
    )
