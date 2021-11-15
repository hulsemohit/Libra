import copy
import random
from typing import List, Optional

import numpy as np
import xxhash


class State:
    def __init__(self, size: int, board: np.ndarray, shapes: List[np.ndarray]):
        self.size = size
        self.board = board
        self.shapes = shapes
        super().__init__()

    def result(self) -> Optional[int]:
        for shape in self.shapes:
            rows, cols = shape.shape
            count = np.sum(shape * shape)
            for i in range(self.size - rows + 1):
                for j in range(self.size - cols + 1):
                    cur = np.sum(self.board[i : i + rows, j : j + cols] * shape)
                    if abs(cur) == count:
                        return cur // count

        if (self.board != 0).all():
            return 0

        return None

    def moves(self) -> np.ndarray:
        return (self.board == 0).flatten().astype(int)

    def move(self, index: int) -> "State":
        new_board = self.board.flatten()
        new_board[index] = 1
        new_board *= -1
        new_board = np.reshape(new_board, (self.size, self.size))
        return State(self.size, new_board, self.shapes)

    def __hash__(self) -> int:
        hasher = xxhash.xxh64()
        hasher.update(self.board)
        return hasher.intdigest()

    def __eq__(self, other: "State") -> bool:
        return (self.board == other.board).all()

    def __str__(self, p=1) -> str:
        return "\n" + "\n".join(
            (
                " ".join(
                    "x" if x * p == 1 else ("o" if x * p == -1 else ".") for x in row
                )
            )
            for row in self.board
        )
