from random import random
from typing import List, Optional

import numpy as np

from mcts import MCTS
from neuralnetwork import NeuralNet
from state import State
from train import train


class Game:
    def __init__(self, size: int, winning_shapes: List[np.ndarray], **kwargs):
        self.args = {
            "iters": 20,
            "episodes": 20,
            "simulations": 20,
            "matches": 20,
            "threshold": 0.55,
        }
        self.args.update(kwargs)
        self.size = size
        self.winning_shapes = winning_shapes
        self.current_player = 1
        self.start = State(size, np.zeros((size, size)), winning_shapes)
        self.state = self.start
        self.nnet = NeuralNet(size)
        self.mcts = MCTS(self.nnet, self.args["simulations"])

    def train(self):
        self.mcts = MCTS(train(self.nnet, self.start, self.args), self.mcts.simulations)

    def reset(self):
        self.current_player = 1
        self.state = self.start

    def predict(self, temperature: float = 1.0) -> int:
        pis = self.mcts.predict_moves(self.state, temperature)
        return np.random.choice(range(len(pis)), p=pis)

    def move(self, index):
        self.state = self.state.move(index)
        self.current_player *= -1

    def evaluation(self) -> float:
        return self.nnet.predict(self.state)[-1]

    def current_board(self) -> np.ndarray:
        return self.state.board * self.current_player

    def result(self) -> Optional[float]:
        res = self.state.result()
        if res is not None:
            return res * self.current_player
        return None

    def __str__(self) -> str:
        return self.state.__str__(self.current_player)
