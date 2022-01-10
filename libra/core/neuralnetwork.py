from typing import List, Tuple

import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from libra.core.state import State

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class NeuralNet:
    def __init__(self, size: int):
        self.input = Input(shape=(size, size))
        x_image = Reshape((size, size, 1))(self.input)

        h_conv1 = Activation("relu")(
            BatchNormalization(axis=3)(Conv2D(256, size, padding="same")(x_image))
        )
        h_conv2 = Activation("relu")(
            BatchNormalization(axis=3)(Conv2D(256, size, padding="same")(h_conv1))
        )
        h_conv3 = Activation("relu")(
            BatchNormalization(axis=3)(Conv2D(256, size, padding="same")(h_conv2))
        )
        h_conv4 = Activation("relu")(
            BatchNormalization(axis=3)(Conv2D(256, size, padding="valid")(h_conv3))
        )
        h_conv4_flat = Flatten()(h_conv4)

        s_fc1 = Dropout(0.3)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(256)(h_conv4_flat)))
        )
        s_fc2 = Dropout(0.3)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(512)(s_fc1)))
        )

        self.pi = Dense(size * size, activation="softmax", name="pi")(s_fc2)
        self.v = Dense(1, activation="tanh", name="v")(s_fc2)

        self.model = Model(inputs=self.input, outputs=[self.pi, self.v])
        self.model.run_eagerly = False
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(0.05),
            run_eagerly=False,
        )

    def train(self, examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        inputs, target_pis, target_vs = list(zip(*examples))
        inputs = np.asarray(inputs)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.model.fit(x=inputs, y=[target_pis, target_vs], batch_size=64, epochs=15)

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        board = np.array(state.board)[np.newaxis, :, :]
        pi, v = self.model.predict_on_batch(board)
        return pi[0], v[0][0]

    def save(self, filename: str):
        self.model.save(filename)
