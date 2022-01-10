"""Contains the Game wrapper class for playing a game against the computer.

Classes:
    Game
"""

from typing import List, Optional

import numpy as np
from tensorflow import keras

from libra.core.mcts import MCTS
from libra.core.neuralnetwork import NeuralNet
from libra.core.state import State
from libra.core.train import train
from libra.core import utils


class Game:
    """A wrapper class for playing a game.

    This class serves as the main interface to the rest of the core module.
    It keeps track of the game state and provides methods to train the computer
    opponent, calculate and make moves, and declare the winner.

    Attributes:
        args: a dictionary of arguments used to configure AlphaZero;
            See the documentation for __init__ for a list;
        size: the size of the game board;
        winning_shapes: a list of shapes which win the game;
        current_player: the player whose turn it is;
        start: the default state of the board;
        nnt: the neural network used by AlphaZero;
        mcts: the Monte-Carlo tree search agent used by AlphaZero.
    """

    def __init__(self, size: int, winning_shapes: List[np.ndarray], **kwargs):
        """Constructor for the game class.

        This sets up the configuration options and loads a pretrained newtork if
        specified.

        Args:
            size: the size of the game board;
            winning_shapes: a list of shapes which win the game;
            kwargs: available keyword arguments are:
                iters: the number of iterations of self-play;
                episodes: the number of games to use in each iteration;
                simulations: the number of searches to use in MCTS;
                matches: the number of matches to play against previous iterations;
                threshold: the minimum win rate needed to accept the new model;
                savefile: the name of the file to load a model from, if any.
                See the code for default values of these arguments.
        """
        self.args = {
            "iters": 10,
            "episodes": 16,
            "simulations": 25,
            "matches": 16,
            "threshold": 0.55,
            "savefile": None,
        }
        self.args.update(kwargs)
        self.size = size
        self.winning_shapes = winning_shapes
        self.current_player = 1
        self.start = State(size, np.zeros((size, size)), winning_shapes)
        self.state = self.start
        self.nnet = NeuralNet(size)
        if self.args["savefile"]:
            self.nnet.model = keras.models.load_model(self.args["savefile"])
        self.mcts = MCTS(self.nnet, self.args["simulations"])

    def train(self):
        """Train the model through self-play."""
        self.mcts = MCTS(train(self.nnet, self.start, self.args), self.mcts.simulations)

    def reset(self):
        """Reset the game back to its initial state."""
        self.current_player = 1
        self.state = self.start

    def predict(self, temperature: float = 0.5) -> int:
        """Use the model to calculate the best move."""
        pis = self.mcts.predict_moves(self.state, temperature)
        return np.random.choice(range(len(pis)), p=pis)

    def can_move(self, index: int) -> bool:
        """Checks if the move at index is valid."""
        return bool(self.state.moves()[index])

    def move(self, index: int) -> bool:
        """Perform a move at index, if valid."""
        if not self.can_move(index):
            utils.warn("Attempted invalid move.")
            utils.debug(
                f"In {self.state.__str__(self.current_player)} at index {index}"
            )
            return False
        self.state = self.state.move(index)
        self.current_player *= -1
        return True

    def save_model(self, filename: str):
        """Save the neural network to filename"""
        self.nnet.save(filename)

    def evaluation(self) -> float:
        """Find the evaluation of the current state."""
        return self.nnet.predict(self.state)[-1] * self.current_player

    def current_board(self) -> np.ndarray:
        """Return the current board as an array made of 0, 1, and -1."""
        return self.state.board * self.current_player

    def result(self) -> Optional[float]:
        """Get the result of the game, or None if not complete"""
        res = self.state.result()
        if res is not None:
            return res * self.current_player
        return None

    def __str__(self) -> str:
        return self.state.__str__(self.current_player)
