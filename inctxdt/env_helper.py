import gymnasium
import numpy as np


class FlattenEnv(gymnasium.ObservationWrapper):
    def __init__(self, env, obs_shape):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=obs_shape)

    def observation(self, obs):
        return obs["observation"]


def get_env_gymnasium(dataset, config):
    base_env = gymnasium.make(dataset.env_name)
    base_obs = base_env.reset()
    observation_shape = base_obs.observation

    needs_flatten = False
    if isinstance(base_obs, dict):
        needs_flatten = True
        base_obs = base_obs["observation"]
        observation_shape = observation_shape.observation

    def fn():
        env = gymnasium.make(dataset.dataset_name)
        env = FlattenEnv(env, base_obs.shape) if needs_flatten else env
        env = gymnasium.wrappers.NormalizeObservation(env)
        env = gymnasium.wrappers.TransformReward(env, lambda rew: rew * config.reward_scale)
        return env

    base_env = fn()
    obs_space = base_env.observation_space

    venv = gymnasium.vector.SyncVectorEnv([fn for _ in range(config.eval_episodes)], observation_space=obs_space)

    return fn, base_env, venv, base_env.observation_space, base_env.action_space


def get_env_gym(dataset, config):
    import gym
    import d4rl

    base_env = gym.make(dataset.dataset_name)
    # base_obs = base_env.reset()

    def fn():
        env = gym.make(dataset.dataset_name)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformReward(env, lambda rew: rew * config.reward_scale)
        return env

    base_env = fn()
    obs_space = base_env.observation_space
    venv = gym.vector.SyncVectorEnv([fn for _ in range(config.eval_episodes)], observation_space=obs_space)

    return fn, base_env, venv, base_env.observation_space, base_env.action_space


def get_env(dataset, config):
    if getattr(dataset, "_d4rl_dataset", False):
        return get_env_gym(dataset, config)
    return get_env_gymnasium(dataset, config)


# terrible function to support both d4rl and gym environments
def _get_env(dataset, config):
    _state_mean = dataset.state_mean.squeeze()
    _state_std = dataset.state_std.squeeze()

    def _obs_transform(obs):
        if isinstance(obs, dict):
            obs = obs["observation"]

        return (obs - _state_mean) / _state_std

    def from_gym(_gym, env_name):
        def fn():
            env_ = _gym.make(env_name)
            obs = env_.reset()[0]

            if isinstance(obs, dict):
                obs = obs["observation"]

            obs_shape = obs.shape
            # env_ = FlattenEnv(env_, obs_shape)
            env_ = _gym.wrappers.NormalizeObservation(env_)
            # env_ = _gym.wrappers.TransformObservation(env_, _obs_transform)
            env_ = _gym.wrappers.TransformReward(env_, lambda rew_: rew_ * config.reward_scale)
            return env_

        env = fn()

        obs_space = env.observation_space
        venv = _gym.vector.SyncVectorEnv([fn for _ in range(config.eval_episodes)], observation_space=obs_space)

        return env, venv

    try:
        import gym
        import d4rl

        env, venv = from_gym(gym, dataset.dataset_name)
        env.seed(config.seed)

    except:
        env, venv = from_gym(gymnasium, dataset.env_name)
    return env, venv


if __name__ == "__main__":
    pass
    # class Dataset:
    #     state_mean = np.array([1, 2, 3])
    #     state_std = np.array([1, 2, 3])

    # dataset = Dataset()

    # env, venv = get_env(dataset, config)
