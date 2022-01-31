from anml.data.component import Component
from anml.data.prototype import DataPrototype
from anml.data.validator import NoNans, Positive, Validator
from numpy.typing import NDArray


class ZeroOneClose(Validator):

    def __call__(self, key: str, value: NDArray):
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"Column {key} contains numbers that are outside "
                             "[0, 1] interval.")


class Data(DataPrototype):

    def __init__(self, obs: str, obs_se: str):
        obs = Component(obs, [NoNans()])
        obs_se = Component(obs_se, [NoNans(), Positive()], default_value=1.0)
        weights = Component(
            "weights", [NoNans(), ZeroOneClose()], default_value=1.0
        )
        components = {
            "obs": obs,
            "obs_se": obs_se,
            "weights": weights,
        }
        super().__init__(components)
