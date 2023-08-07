from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from distutils.util import strtobool


_warned_attrs = set()


def _get_env_spec(env_name: str) -> Tuple[int, int]:
    import gym  # might need gymnasium

    env = gym.make(env_name)

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    return action_dim, state_dim


@dataclass
class EnvSpec:
    episode_len: int
    seq_len: int
    env_name: str

    action_dim: int = None
    state_dim: int = None

    def __post_init__(self):
        if (self.action_dim is None) or (self.state_dim is None):
            self.action_dim, self.state_dim = _get_env_spec(self.env_name)

    # @classmethod


@dataclass
class config_tool:
    env_name: str = "pointmaze-umaze-v1"
    device: str = "cpu"

    epochs: int = 1
    num_workers: int = 8
    batch_size: int = 32
    n_batches: int = -1

    # dataset
    seq_len: int = 20
    episode_len: int = 1000

    num_layers: int = 6
    num_heads: int = 8

    adist: bool = False  # accelerate distributed
    dist: bool = False  # pytorch distributed

    # optim
    betas: Tuple[float, float] = (0.9, 0.999)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100

    clip_grad: Optional[float] = 0.25

    # eval
    reward_scale: float = 0.001
    eval_episodes: int = 5

    _debug: bool = False
    debug_note: str = ""
    __singleton = None

    def __post_init__(self):
        pass

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]

        if (attr not in _warned_attrs) and self._debug:
            print(f" WARNING: Used non-existent config field:`{str(attr)}`. Returning None.")
            _warned_attrs.add(attr)

        return None

    def get_env_spec(self, env_name: str = None):
        env_name = env_name or self.env_name

        return EnvSpec(
            episode_len=self.episode_len,
            seq_len=self.seq_len,
            env_name=env_name,
            # action_dim=action_dim,
            # state_dim=state_dim,
        )

    @classmethod
    def get(cls, **kwargs):
        """
        helper class to make singleton config object
        """
        # allow passed in kwargs to override
        for k, v in kwargs.items():
            setattr(cls, k, v)

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
                    elif val.lower() in (
                        "true",
                        "false",
                    ):
                        val = bool(strtobool(val))

                    parser.add_argument(f"--{field}", default=val, type=type(val))
            args, extra = parser.parse_known_args()

        for k, v in vars(args).items():
            setattr(cls, k, v)

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cls.device == "cpu":
            print("WARNING RUNNING IN CPU MODE.  ONLY FOR DEV.")

        if cls.__singleton is None:  # might want this before arg related
            cls.__singleton = cls()

        _debug_note(cls.debug_note.upper())

        return cls.__singleton


def _debug_note(note: str = None):
    if note in ["", None]:
        return

    import atexit

    def _print_info():
        print("===" * 30)
        print("\t--RUN/DEBUG-NOTE:")
        print("\n\n==>🤠|>", note.upper(), "<|🤠<==\n\n")
        print("===" * 30)

    atexit.register(_print_info)
    _print_info()


if __name__ == "__main__":
    conf = config_tool.get()
    val = conf.asdfl
    breakpoint()
