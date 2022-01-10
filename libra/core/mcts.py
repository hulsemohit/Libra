import random

import numpy as np

from libra.core.neuralnetwork import NeuralNet
from libra.core.state import State
from libra.core import utils


class MCTS:
    def __init__(self, nnet: NeuralNet, simulations: int):
        self.nnet = nnet
        self.move_visits = {}
        self.state_visits = {}
        self.valuation = {}
        self.policy = {}
        self.moves = {}
        self.result = {}
        self.simulations = simulations

    def predict_moves(self, state: State, temp: float = 1) -> np.ndarray:
        for _ in range(self.simulations):
            self.search(state)

        counts = np.array(
            [
                float(self.move_visits.get((state, a), 0))
                for a in range(len(self.moves[state]))
            ]
        )

        if temp == 0:
            best = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best = best[random.randrange(len(best))]
            probs = [0.0] * len(counts)
            probs[best] = 1.0
            return np.array(probs)

        counts **= 1.0 / temp
        counts /= sum(counts)
        return counts

    def search(self, state: State) -> float:
        if state not in self.result:
            self.result[state] = state.result()

        if self.result[state] is not None:
            return self.result[state]

        if state not in self.policy:
            self.policy[state], value = self.nnet.predict(state)
            self.moves[state] = state.moves()
            self.policy[state] = self.policy[state] * self.moves[state]
            if np.sum(self.policy[state]) != 0:
                self.policy[state] /= np.sum(self.policy[state])
            self.state_visits[state] = 0
            return value

        def evaluation(move):
            if self.moves[state][move] == 0:
                return -np.Infinity

            val = self.valuation.get((state, move), 0)
            vis = self.move_visits.get((state, move), 0)
            tot = self.state_visits[state]
            policy = self.policy[state][move]

            if (
                tot < 0
                or (1 + vis) <= 0
                or np.isnan(val)
                or np.isnan(vis)
                or np.isnan(tot)
                or np.isnan(policy)
            ):
                utils.warn(f"Received invalid information for state {str(state)}.")
                utils.debug(
                    f"val = {val}, vis = {vis}, tot = {tot}, policy = {policy}."
                )
                return -np.Infinity

            return val + policy * (2 * tot / (1 + vis)) ** 0.5

        best_move = max(range(len(self.moves[state])), key=evaluation)

        if evaluation(best_move) == -np.Infinity:
            utils.warn(
                f"No valid moves found for {state} (state.moves is {state.moves()}), choosing {best_move}."
            )
            utils.debug(
                f"Evaluations are {list(map(evaluation, range(len(state.moves()))))}."
            )

        value = -self.search(state.move(best_move))

        self.state_visits[state] += 1
        key = (state, best_move)
        if (state, best_move) in self.move_visits:
            self.valuation[key] *= self.move_visits[key]
            self.valuation[key] += value
            self.move_visits[key] += 1
            self.valuation[key] /= self.move_visits[key]
        else:
            self.valuation[key] = value
            self.move_visits[key] = 1

        return value
