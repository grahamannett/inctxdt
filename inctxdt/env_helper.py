import gymnasium
import numpy as np


class FlattenEnv(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_shape):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=obs_shape)

    def observation(self, obs):
        return obs["observation"]


def get_env_gymnasium(dataset, config, make_venv: bool = True):
    base_env = gymnasium.make(dataset.env_name)
    base_obs = base_env.reset()[0]

    observation_shape = base_env.observation_space

    needs_flatten = False
    if isinstance(base_obs, dict):
        needs_flatten = True
        base_obs = base_obs["observation"]
        observation_shape = observation_shape["observation"]

    def fn():
        env = gymnasium.make(dataset.env_name)
        env = FlattenEnv(env, base_obs.shape) if needs_flatten else env
        env = gymnasium.wrappers.NormalizeObservation(env)
        env = gymnasium.wrappers.TransformReward(env, lambda rew: rew * config.reward_scale)
        return env

    base_env = fn()
    obs_space = base_env.observation_space

    out = (
        fn,
        base_env,
    )

    if make_venv:
        venv = gymnasium.vector.SyncVectorEnv([fn for _ in range(config.eval_episodes)], observation_space=obs_space)
        out += (venv,)

    return out + (observation_shape, base_env.action_space)


def get_env_gym(dataset, config, make_venv: bool = True):
    import gym
    import d4rl

    base_env = gym.make(dataset.dataset_name)
    # base_obs = base_env.reset()[0]

    def fn():
        env = gym.make(dataset.dataset_name)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformReward(env, lambda rew: rew * config.reward_scale)
        return env

    base_env = fn()
    obs_space = base_env.observation_space

    out = (fn, base_env)

    if make_venv:
        venv = gym.vector.SyncVectorEnv([fn for _ in range(config.eval_episodes)], observation_space=obs_space)
        out += (venv,)

    return out + (obs_space, base_env.action_space)


def get_env(dataset, config):
    if getattr(dataset, "_d4rl_dataset", False):
        return get_env_gym(dataset, config)
    return get_env_gymnasium(dataset, config)


if __name__ == "__main__":
    pass
    # class Dataset:
    #     state_mean = np.array([1, 2, 3])
    #     state_std = np.array([1, 2, 3])

    # dataset = Dataset()

    # env, venv = get_env(dataset, config)
