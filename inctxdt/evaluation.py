from typing import Any, Dict, Tuple

import numpy as np
import torch

# from gym import Env
# from gymnasium import Env
# from gym import Env
# from gym import vector
import gym

from inctxdt.config import EnvSpec, config_tool

from inctxdt.model import DecisionTransformer
from inctxdt.model_output import ModelOutput


def fix_obs_dict(obs_dict: Dict[str, np.array] | np.array) -> np.array:
    if not isinstance(obs_dict, dict):
        return obs_dict

    return np.concatenate([obs_dict[key] for key in obs_dict.keys()], axis=-1)


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    env_spec: EnvSpec,
    device: str,
    target_return: float,
) -> Tuple[float, float]:
    model.eval()
    states = torch.zeros(1, env_spec.episode_len + 1, env_spec.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(1, env_spec.episode_len, env_spec.action_dim, dtype=torch.float, device=device)
    returns = torch.zeros(1, env_spec.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(env_spec.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states_init = env.reset()

    states_init = fix_obs_dict(states_init)

    states[:, 0] = torch.as_tensor(states_init, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    terminated, truncated = False, False
    episode_return, episode_len = 0.0, 0.0
    for step in range(env_spec.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)

        output = model(  # fix this noqa!!!
            states[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            actions[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            returns[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            time_steps[:, : step + 1][:, -env_spec.seq_len :],  # noqa
        )
        logits = output.logits

        assert (logits.shape[0] == 1) or (len(logits.shape) > 2), "batch size must be 1 for evaluation"

        predicted_action = logits[0, -env_spec.action_dim :].squeeze().cpu().numpy()

        # unpack
        next_state, reward, *done, info = env.step(predicted_action)

        if len(done) >= 2:  # NOTE: there will be truncated and terminated later - throw error and catch this
            assert False, "not sure if i handle these correctly"  #
        done = done[0]  #  should be= done[0] if len(done) == 1 else (done[0] or done[1])

        next_state = fix_obs_dict(next_state)

        # next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len


def _check_states(states: Any) -> TypeError:
    assert isinstance(states, np.ndarray), "states must be a numpy array. check wrapper."


@torch.no_grad()
def venv_eval_rollout(
    model: DecisionTransformer,
    venv: gym.vector.VectorEnv,
    env_spec: EnvSpec,
    device: str,
    target_return: float,
) -> Tuple[float, float]:
    num_envs = venv.num_envs
    episode_len = env_spec.episode_len
    seq_len = env_spec.seq_len
    model.eval()

    states = torch.zeros(num_envs, episode_len + 1, env_spec.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(num_envs, episode_len, env_spec.action_dim, dtype=torch.float, device=device)
    returns = torch.zeros(num_envs, episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(env_spec.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.repeat(num_envs, 1).view(num_envs, -1)

    states_init = venv.reset()

    _check_states(states_init)  # # states_init = fix_obs_dict(states_init)

    states[:, 0] = torch.as_tensor(states_init, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    terminated, truncated = False, False

    # float64 b/c rewards come back as float64 and faster to not convert every time
    venv_episode_return = torch.zeros(num_envs, dtype=torch.float64)
    venv_episode_len = torch.zeros(num_envs, dtype=torch.int)
    dones = torch.zeros(num_envs, dtype=torch.bool)

    for step in range(episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)

        output = model(
            states[~dones, : step + 1][:, -seq_len:],  # noqa
            actions[~dones, : step + 1][:, -seq_len:],  # noqa
            returns[~dones, : step + 1][:, -seq_len:],  # noqa
            time_steps[~dones, : step + 1][:, -seq_len:],  # noqa
        )

        # predicted_action = output.logits.squeeze()
        # predicted_action = predicted_action[~dones, -env_spec.action_dim :].cpu().numpy()
        # predicted_action = output.logits[~dones, -env_spec.action_dim :].squeeze().cpu().numpy()
        predicted_action = output.logits.reshape(venv.num_envs, -1)
        predicted_action = predicted_action[~dones, -env_spec.action_dim :].squeeze().cpu().numpy()

        # unpack
        next_state, rew, *step_dones, info = venv.step(predicted_action)

        # NOTE: there will be truncated and terminated later - throw error and catch this
        assert len(step_dones) == 1, "not sure if i handle these correctly. need to handle terminated/truncated"

        dones[step_dones[0]] = True

        # next_state = fix_obs_dict(next_state)

        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[~dones, step] = torch.as_tensor(predicted_action[~dones], device=device)
        states[~dones, step + 1] = torch.as_tensor(next_state[~dones], device=device, dtype=torch.float)
        returns[~dones, step + 1] = returns[~dones, step] - torch.as_tensor(rew[~dones], dtype=torch.float).to(device)

        venv_episode_return[~dones] += rew[~dones]
        venv_episode_len += ~dones  # or (~dones).to(int)

        if (dones == True).all():
            venv_episode_len += 1  # account for last step
            break

    return venv_episode_return, venv_episode_len


if __name__ == "__main__":
    import d4rl

    model = DecisionTransformer(17, 6, 20)
    env_spec = EnvSpec(episode_len=1000, seq_len=20, action_dim=6, state_dim=17)
    env = gym.vector.make("halfcheetah-medium-v2", num_envs=10, asynchronous=False)

    vals = venv_eval_rollout(model, env, env_spec, "cpu", 12000.0)
    print(vals)
