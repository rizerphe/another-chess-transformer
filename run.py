import chess
import torch
import torch.nn.functional as F

from board_to_tensors import board_to_tensors
from model import Config, Transformer


board = chess.Board("1B1Q2k1/5p1p/6p1/p3p3/4P3/1B6/PPP1NPPP/R3K2R b KQ - 0 23")
if not board.is_valid():
    print("Invalid board")
    exit(1)
pieces, positions, players = board_to_tensors(board)

transformer = Transformer(Config(n_layers=6, n_heads=4, d_model=16))
transformer.load_state_dict(torch.load("transformer.pth"))

piece_logits, target_logits = transformer(
    pieces.unsqueeze(0), positions.unsqueeze(0), players.unsqueeze(0)
)

piece_probs = F.softmax(piece_logits, dim=1)

# Now, we extract the sorted pieces to move
top_pieces = piece_probs[0].topk(len(pieces.tolist())).indices

# We find the first piece that actually exists on the board
# and belongs to the player to move
for piece in top_pieces:
    if piece in list(positions):
        top_piece = positions.tolist().index(piece)
        top_piece_coord = piece

        # Check if the piece belongs to the player to move
        player_to_move = 1 if board.turn == chess.WHITE else 0
        if players[top_piece] == player_to_move:
            break

# Now, we extract the target probs for the top piece
target_probs = F.softmax(target_logits, dim=2)
top_target = target_probs[0, top_piece].topk(1).indices.item()

# We get the coordinates of each square in chess terms, not as 0-63
source_square = chess.square_name(top_piece_coord)
target_square = chess.square_name(top_target)

print(board)
print(f"Move: {source_square} -> {target_square}")
print(board.fen())
