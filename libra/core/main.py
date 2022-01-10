import numpy as np

from libra.core.game import Game
from libra.core import utils

TTT_WINNING_SHAPES = [
    [
        [1, 1, 1],
    ],
    [
        [1],
        [1],
        [1],
    ],
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ],
]

WINNING_SHAPES_L = [
    [
        [0, 1],
        [1, 1],
    ],
    [
        [1, 0],
        [1, 1],
    ],
    [
        [1, 1],
        [0, 1],
    ],
    [
        [1, 1],
        [1, 0],
    ],
]

WINNING_SHAPES_3LINE = [
    [
        [1, 1, 1],
    ],
    [
        [1],
        [1],
        [1],
    ],
]

WINNING_SHAPES_PLUS = [
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ],
]

BOARD_SIZE = 3

EMPTY_BOARD = np.zeros((BOARD_SIZE, BOARD_SIZE))


def main():
    utils.clear()
    game = Game(BOARD_SIZE, [np.array(x) for x in TTT_WINNING_SHAPES], iters=5)
    game.train()

    while True:
        while game.state.result() is None:
            if game.current_player == 1:
                game.move(game.predict())
            else:
                game.move(int(input("Move: ")))
            print(str(game))
        print(game.result())
        game.reset()


if __name__ == "__main__":
    main()
