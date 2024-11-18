import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class MultiHeadAttention(nn.Module):
    """A multi-head attention layer."""

    def __init__(self, config: Config) -> None:
        """Initialize the layer.

        Args:
            config: The configuration for the model.
        """
        super().__init__()

        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)

        self.output = nn.Linear(config.d_model, config.d_model)

        self.config = config

    def _split_heads(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return x.view(
            -1, seq_len, self.config.n_heads, self.config.d_embedding
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        q = self._split_heads(self.q(x), seq_len)
        k = self._split_heads(self.k(x), seq_len)
        v = self._split_heads(self.v(x), seq_len)

        att = (q @ k.transpose(-2, -1)) * (self.config.d_embedding**-0.5)
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(*x.size())

        return self.output(y)
