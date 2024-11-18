import torch
import yaml

import chess


def board_to_tensors(board: chess.Board):
    # Get a list of pieces along with their positions and players
    possible_pieces = "PNBRQK"

    pieces = []
    positions = []
    players = []
    for square, piece in board.piece_map().items():
        pieces.append(possible_pieces.index(piece.symbol().upper()))
        positions.append(square)
        players.append(int(piece.color == chess.WHITE))

    # Convert the lists to tensors
    pieces = torch.tensor(pieces)
    positions = torch.tensor(positions)
    players = torch.tensor(players)

    return pieces, positions, players


def iterate_over_dataset(path="fens.yaml"):
    tactics = yaml.safe_load(open(path, "r"))

    for board, moves in tactics.items():
        board = chess.Board(board)
        move = board.parse_san(moves[0])

        pieces, positions, players = board_to_tensors(board)

        # Get source square coordinate
        source_square = move.from_square
        # Get target square coordinate
        target_square = move.to_square

        # Convert them to a one-hot tensor
        piece_targets = torch.zeros(64)
        piece_targets[source_square] = 1
        target_targets = torch.zeros(64)
        target_targets[target_square] = 1

        # Get a piece_to_move tensor: of length seq_len, with the source square repeated
        piece_to_move = torch.tensor(
            [1 if square == source_square else 0 for square in positions]
        )

        yield pieces, positions, players, piece_targets, target_targets, piece_to_move
