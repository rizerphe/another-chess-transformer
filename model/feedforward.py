import torch
import torch.nn as nn
from transformers.activations import NewGELUActivation

from .config import Config


class FeedForward(nn.Module):
    """A feedforward layer for the transformer."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.upprojection = nn.Linear(config.d_model, 4 * config.d_model)
        self.activation = NewGELUActivation()
        self.downprojection = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.downprojection(self.activation(self.upprojection(e)))
