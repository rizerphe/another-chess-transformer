import time

import torch

from board_to_tensors import iterate_over_dataset
from model import Config, Transformer

transformer = Transformer(Config(n_layers=6, n_heads=4, d_model=16))
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)

losses = []
optimizer.zero_grad()

for i, (
    pieces,
    positions,
    players,
    piece_targets,
    target_targets,
    piece_to_move,
) in enumerate(iterate_over_dataset("fens_smol.yaml")):
    loss = transformer(
        pieces.unsqueeze(0),
        positions.unsqueeze(0),
        players.unsqueeze(0),
        piece_targets.unsqueeze(0),
        target_targets.unsqueeze(0),
        piece_to_move.unsqueeze(0),
    )
    loss.backward()

    losses.append(loss.item())

    if i % 100 == 0:
        optimizer.step()
        print(f"Step {i}, loss: {sum(losses) / len(losses)}")
        losses = []
        optimizer.zero_grad()

# Save the model
torch.save(transformer.state_dict(), "transformer.pth")
