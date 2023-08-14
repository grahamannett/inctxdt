
    @classmethod
    def add_commands(cls, default_cmd: callable, other_cmds=[]):
        breakpoint()

        def _wrapper(self, *args, **kwargs):
            return default_cmd(*args, **kwargs)

        cls.cmd: callable = _wrapper
        return cls

        # cls.__annotations__[name] = type
        # setattr(cls, name, default)
        # return cls


class Runner:
    def __new__(cls, comm)

# if __name__ == "__main__":
#     breakpoint()


def train():
    # breakpoint()
    print("hello world")


def baseline():
    print("other")


class Commands:
    train = train
    baseline = baseline

from typing import Callable


import pyrallis


@pyrallis.wrap()
def main(config: Config):
    config.cmd()
    breakpoint()


from dataclasses import dataclass, make_dataclass, field


class Runner:
    def __new__(cls, *args, **kwargs):
        # breakpoint()
        Base = args[0]
        # config.cmd = {
        #     "train": run_autoregressive,
        #     "baseline": run_baseline,
        # }
        # _fields = []
        # for
        # breakpoint()

        _config = make_dataclass("Config", fields=[("cmd", Callable, cls)], bases=(Base,))
        return _config


# class Cmd(Runner):
#     train = run_autoregressive
#     baseline = run_baseline



if __name__ == "__main__":
    Runner(Commands)
    Config.add_commands(train, [baseline])
    main()


