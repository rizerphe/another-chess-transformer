import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .config import Config


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.piece_embeddings = nn.Embedding(6, config.d_model)
        self.player_embeddings = nn.Embedding(2, config.d_model)
        self.position_embeddings = nn.Embedding(64, config.d_model)
        self.tower = nn.Sequential(
            *(TransformerBlock(config) for _ in range(config.n_layers))
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.start_unembedding = nn.Linear(config.d_model, 64, bias=False)
        self.target_unembedding = nn.Linear(config.d_model, 64, bias=False)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        pieces: torch.Tensor,
        positions: torch.Tensor,
        players: torch.Tensor,
        piece_targets: torch.Tensor | None = None,
        target_targets: torch.Tensor | None = None,
        piece_to_move: torch.Tensor | None = None,
    ):
        e = (
            self.piece_embeddings(pieces)
            + self.position_embeddings(positions)
            + self.player_embeddings(players)
        )
        e = self.tower(e)
        e = self.ln_f(e)

        piece_logits = self.start_unembedding(e)
        target_logits = self.target_unembedding(e)

        # Average the piece logits over the sequence dimension
        piece_logits = piece_logits.mean(dim=1)

        if piece_targets is None or target_targets is None or piece_to_move is None:
            return piece_logits, target_logits

        # target_logits is of shape (batch_size, seq_len, 64)
        # piece_to_move is of shape (batch_size, seq_len)
        # We want to multiply them, zeroing out the logits for all squares except the source square
        target_logits = target_logits * piece_to_move.unsqueeze(2)

        # Now, we want to sum them into (batch_size, 64) to be able to calculate the loss
        target_logits = target_logits.sum(dim=1)

        piece_loss = F.cross_entropy(piece_logits, piece_targets)
        target_loss = F.cross_entropy(target_logits, target_targets)

        return piece_loss + target_loss
