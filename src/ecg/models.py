"""Keras architectures.

The MLP is the model the thesis shipped: its layer configuration was read back
out of ``base_models/model_cGAN10k_MLP.h5`` to confirm which of the five
``mlp_model`` variants scattered through ``validation.ipynb`` produced the saved
weights. It is the one without dropout or L2.
"""

from __future__ import annotations

from .config import DEFAULT, Config


def _keras():
    import keras

    return keras


def mlp(
    input_size: int | None = None,
    n_classes: int | None = None,
    hidden: tuple[int, ...] = (128, 64),
    cfg: Config = DEFAULT,
):
    """Dense classifier: ``input_size -> 128 -> 64 -> n_classes``.

    With the defaults this is 27,972 parameters -- small enough to be the basis
    of the thesis' low-latency argument.
    """
    keras = _keras()
    input_size = cfg.window_length if input_size is None else input_size
    n_classes = cfg.n_classes if n_classes is None else n_classes

    layers = [keras.layers.Input(shape=(input_size,))]
    layers += [keras.layers.Dense(units, activation="relu") for units in hidden]
    layers.append(keras.layers.Dense(n_classes, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def cnn(
    input_size: int | None = None,
    n_classes: int | None = None,
    filters: tuple[int, ...] = (32, 64),
    kernel_size: int = 5,
    pool_size: int = 2,
    cfg: Config = DEFAULT,
):
    """1-D convolutional classifier over the raw beat window."""
    keras = _keras()
    input_size = cfg.window_length if input_size is None else input_size
    n_classes = cfg.n_classes if n_classes is None else n_classes

    layers = [keras.layers.Input(shape=(input_size, 1))]
    for n_filters in filters:
        layers += [
            keras.layers.Conv1D(n_filters, kernel_size, activation="relu",
                                padding="same"),
            keras.layers.MaxPooling1D(pool_size),
        ]
    layers += [
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ]

    model = keras.Sequential(layers)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


ARCHITECTURES = {"mlp": mlp, "cnn": cnn}


def build(name: str, **kwargs):
    """Build an architecture by name."""
    try:
        return ARCHITECTURES[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"unknown architecture {name!r}; expected one of {sorted(ARCHITECTURES)}"
        ) from None
