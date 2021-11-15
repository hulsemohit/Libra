import copy
from typing import List, Tuple, Dict

import numpy as np

from mcts import MCTS
from neuralnetwork import NeuralNet
from state import State
import utils


def train(nnet: NeuralNet, start_state: State, args: Dict) -> NeuralNet:
    examples = []
    for i in range(args["iters"]):
        utils.info(f"Iteration {i + 1}/{args['iters']}")

        utils.info("Genrating examples...")
        for j in range(args["episodes"]):
            examples += generate_examples(MCTS(nnet, args["simulations"]), start_state)
            utils.info(f"Completed {j + 1}/{args['episodes']} sample games.")

        utils.info(f"Training on {len(examples)} examples.")
        newnet = copy.deepcopy(nnet)
        newnet.train(examples)

        utils.info("Testing against previous model.")
        win_rate = test(
            MCTS(newnet, args["simulations"]),
            MCTS(nnet, args["simulations"]),
            start_state,
            args["matches"],
        )

        if win_rate >= args["threshold"]:
            utils.info(f"Accepting new model (win-rate {win_rate})")
            nnet = newnet
        else:
            utils.info(f"Rejecting new model (win-rate {win_rate})")

        random_rate = test(
            MCTS(nnet, args["simulations"]),
            MCTS(NeuralNet(start_state.size), args["simulations"]),
            start_state,
            args["matches"],
        )
        utils.info(f"Current win-rate against random player: {random_rate}.\n")
    return nnet


def generate_examples(
    mcts: MCTS, start: State
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    examples = []
    state = start

    while True:
        res = state.result()
        if res is not None:
            examples.reverse()
            examples = [
                (s, p, np.array([(-1) ** (i + 1) * float(res)]))
                for (i, (s, p)) in enumerate(examples)
            ]
            return symmetries(examples)
        move_pi = mcts.predict_moves(state)
        examples.append((state.board, move_pi))
        state = state.move(np.random.choice(range(len(move_pi)), p=move_pi))


def symmetries(
    examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    size = examples[0][0].shape[0]

    def transform(example, func):
        board, pis, res = example
        pis = pis.reshape((size, size))
        board = func(board)
        pis = func(pis).flatten()
        return board, pis, res

    results = []
    for example in examples:
        for _ in range(2):
            for _ in range(4):
                results.append(transform(example, np.rot90))
            example = transform(example, np.fliplr)
    return results


def test(player1: MCTS, player2: MCTS, start: State, matches: int) -> float:
    score = 0.5 * matches
    mcts = [player1, player2]
    for i in range(matches):
        state = start
        current = 1
        while True:
            utils.debug(f"Player: {current}. {state.__str__(current)}")
            res = state.result()
            if res is not None:
                res = current * res * (-1) ** i
                utils.info(f"Match {i + 1}/{matches} result: {res}")
                score += current * res * 0.5
                break
            move_pi = mcts[current].predict_moves(state)
            state = state.move(np.random.choice(range(len(move_pi)), p=move_pi))
            current *= -1
        mcts = [player2, player1]

    return score / matches
