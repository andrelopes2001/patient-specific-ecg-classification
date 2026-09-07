"""Conditional GAN for minority-class beat synthesis (PyTorch).

Trained per class to synthesise extra S, V and F beats, addressing the 89%/0.8%
imbalance between N and F. Architecture as follows: noise and one-hot
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


class _MinMax:
    """Scale beats into [0, 1] and back, using percentiles rather than extremes.

    The generator ends in a sigmoid, so it can only emit values in [0, 1] --
    while raw beats are in millivolts and frequently negative. Some scaling is
    therefore mandatory.

    Scaling by the true min and max is what the obvious implementation does, and
    it does not work: beat amplitudes have long tails (-3.2 to +3.9 mV, but the
    1st-99th percentile is only [-0.77, +1.85]), so real beats end up occupying
    37% of the output range with a standard deviation of 0.06. An untrained
    generator emits across the whole interval, the two distributions barely
    overlap in scale, and *neither* network learns -- both losses sit at ln(2)
    indefinitely.

    Clipping at percentiles spreads the real beats across the sigmoid's range.
    Measured effect on generated-beat quality (mean absolute deviation from the
    real per-class mean beat, lower is better; 0.146 is what a constant scores):

        global min-max   0.865   discriminator loss 0.693 (chance)
        0.5-99.5 pct     0.339   discriminator loss 0.638
        2-98 pct         0.192   discriminator loss 0.595

    The cost is clipping: beats outside the percentile range saturate.
    """

    def __init__(self, lo_pct: float = 2.0, hi_pct: float = 98.0):
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct

    def fit(self, X: np.ndarray) -> "_MinMax":
        self.lo = float(np.percentile(X, self.lo_pct))
        self.hi = float(np.percentile(X, self.hi_pct))
        self.span = max(self.hi - self.lo, 1e-12)
        return self

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.lo) / self.span, 0.0, 1.0)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * self.span + self.lo


def train_cgan(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config = DEFAULT,
    epochs: int = 20_000,
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    device: str | torch.device = "cpu",
    verbose: bool = False,
):
    """Train the conditional GAN on real beats.

    Standard alternating GAN training with BCE loss: the discriminator is
    updated on a real and a fake batch, then the generator is updated to fool
    it. Returns ``(generator, scaler, losses)``.

    ``epochs`` counts minibatch steps, not passes over the data. Adam at 2e-4
    with beta1 0.5 is the usual GAN setting; SGD at 0.1 (which an earlier
    version used) does not converge here at any step count tried.
    """
    torch.manual_seed(cfg.seed)
    scaler = _MinMax().fit(X)
    Xs = scaler.forward(X).astype(np.float32)

    generator, discriminator = build(cfg, device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=learning_rate,
                             betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,
                             betas=(0.5, 0.999))
    schedulers = []
    bce = nn.BCELoss()
    rng = torch.Generator().manual_seed(cfg.seed)
    losses = {"generator": [], "discriminator": []}

    ones = torch.ones(batch_size, device=device)
    zeros = torch.zeros(batch_size, device=device)

    for epoch in range(epochs):
        # --- discriminator: real beats are 1, generated beats are 0 ---
        real, context = minibatch(Xs, y, batch_size, device, rng, cfg.n_classes)
        if len(real) < batch_size:
            continue
        noise = sample_noise(batch_size, generator.latent_dim, device)
        fake = generator(noise, context).detach()

        loss_d = 0.5 * (
            bce(discriminator(real, context).reshape(batch_size), ones)
            + bce(discriminator(fake, context).reshape(batch_size), zeros)
        )
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # --- generator: try to make the discriminator answer 1 ---
        noise = sample_noise(batch_size, generator.latent_dim, device)
        _, context = minibatch(Xs, y, batch_size, device, rng, cfg.n_classes)
        loss_g = bce(
            discriminator(generator(noise, context), context).reshape(batch_size), ones
        )
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        for scheduler in schedulers:
            scheduler.step()

        losses["discriminator"].append(float(loss_d))
        losses["generator"].append(float(loss_g))
        if verbose and (epoch + 1) % 5000 == 0:
            print(f"    epoch {epoch + 1}: D={float(loss_d):.3f} G={float(loss_g):.3f}",
                  flush=True)

    return generator, scaler, losses


def synthesise(
    generator: Generator,
    scaler: _MinMax,
    labels: np.ndarray,
    device: str | torch.device = "cpu",
    n_classes: int = len(CLASSES),
) -> np.ndarray:
    """Generate beats for the given class labels, in the original units."""
    return scaler.inverse(generate(generator, labels, device, n_classes))
