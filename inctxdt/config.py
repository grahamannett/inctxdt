from typing import NamedTuple, Tuple
import torch

_warned_attrs = set()


class EnvSpec(NamedTuple):
    action_dim: int
    state_dim: int

    episode_len: int
    seq_len: int


class config_tool:
    env_name: str = "pointmaze-umaze-v1"
    device: str = "cpu"

    epochs: int = 1
    batch_size: int = 4

    dist: bool = False  # pytorch distributed
    adist: bool = False  # accelerate distributed

    # optim
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    warmup_steps: int = 10

    clip_grad: bool = True

    _debug: bool = False
    __singleton = None

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]

        if (attr not in _warned_attrs) and self._debug:
            print(
                f" WARNING: Used non-existent config field:`{str(attr)}`. Returning None."
            )
            _warned_attrs.add(attr)

        return None

    @classmethod
    def get(cls):
        import argparse

        parser = argparse.ArgumentParser()
        for k, v in cls.__dict__.items():
            if k.startswith("_"):
                continue
            parser.add_argument(f"--{k}", type=type(v), default=v)
        args, extra = parser.parse_known_args()

        if extra:
            for e in extra:
                if e.startswith("--") and "=" in e:
                    field, val = e.split("=")
                    field = field[2:]
                    if val.isdigit():
                        val = float(val)
                    parser.add_argument(f"--{field}", default=val, type=type(val))
            args, extra = parser.parse_known_args()

        for k, v in vars(args).items():
            setattr(cls, k, v)

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cls.device == "cpu":
            print("WARNING RUNNING IN CPU MODE.  ONLY FOR DEV.")

        if cls.__singleton is None:  # might want this before arg related
            cls.__singleton = cls()

        return cls.__singleton


if __name__ == "__main__":
    conf = config_tool.get()
    val = conf.asdfl
    breakpoint()
