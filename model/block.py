import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .config import Config
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """A single simple transformer block with a multi-head attention"""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)

        self.attn = MultiHeadAttention(config)
        self.feedforward = FeedForward(config)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        e = e + self.attn(self.ln_1(e))
        e = e + self.feedforward(self.ln_2(e))
        return e
