from typing import Any, Dict, Tuple

import gymnasium
import numpy as np
import torch
from gym.vector import VectorEnv
from tqdm.auto import trange

from inctxdt.config import EnvSpec
from inctxdt.episode_data import EpisodeData
from inctxdt.models.model import DecisionTransformer
from inctxdt.models.model_output import ModelOutput


def fix_obs_dict(obs: Dict[str, np.array] | np.array) -> np.array:
    if isinstance(obs, dict):
        # np.concatenate([obs_dict[key] for key in obs_dict.keys()], axis=-1)
        obs = obs["observation"]
    return obs


def _check_states(states: Any) -> TypeError:
    assert isinstance(states, np.ndarray), "states must be a numpy array. check wrapper."


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: torch.nn.Module,
    env: gymnasium.Env,
    env_spec: EnvSpec,
    device: str,
    target_return: float,
) -> Tuple[float, float]:
    model.eval()
    states = torch.zeros(1, env_spec.episode_len + 1, env_spec.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(1, env_spec.episode_len, env_spec.action_dim, dtype=torch.float, device=device)
    returns = torch.zeros(1, env_spec.episode_len + 1, dtype=torch.float, device=device)
    timesteps = torch.arange(env_spec.episode_len, dtype=torch.long, device=device).view(1, -1)

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

        output = model(
            states=states[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            actions=actions[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            returns_to_go=returns[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            timesteps=timesteps[:, : step + 1][:, -env_spec.seq_len :],  # noqa
        )

        logits = output.logits if hasattr(output, "logits") else output

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


@torch.no_grad()
def venv_eval_rollout(
    model: torch.nn.Module,
    venv: gymnasium.vector.VectorEnv,
    env_spec: EnvSpec,
    device: str,
    target_return: float,
    output_sequential: bool = False,
    prior_episode: EpisodeData = None,
) -> Tuple[float, float]:
    assert isinstance(venv, (gymnasium.vector.VectorEnv, VectorEnv)), "venv must be a vectorized env."

    model.eval()
    num_envs = venv.num_envs
    # seq len is the model seq len, i.e. context length
    seq_len = env_spec.seq_len
    max_episode_len = env_spec.episode_len

    # episode len is the length of the data that we will at max run for plus the prior episode
    episode_len = max_episode_len if prior_episode is None else len(prior_episode)
    # init_idx is the index where the eval begins in the states/actions/returns.  0 if no prior episode
    init_idx = 0

    states = torch.zeros(num_envs, episode_len + 1, env_spec.state_dim, dtype=torch.float, device=device)
    returns = torch.zeros(num_envs, episode_len + 1, dtype=torch.float, device=device)
    actions = torch.zeros(num_envs, episode_len, env_spec.action_dim, dtype=torch.float, device=device)
    timesteps = torch.arange(env_spec.episode_len, dtype=torch.long, device=device)
    timesteps = timesteps.repeat(num_envs, 1).view(num_envs, -1)

    if prior_episode:
        states[:, : len(prior_episode)] = torch.as_tensor(prior_episode.states, device=device)
        actions[:, : len(prior_episode)] = torch.as_tensor(prior_episode.actions, device=device)
        returns[:, : len(prior_episode)] = torch.as_tensor(prior_episode.returns_to_go, device=device)
        prior_timesteps = torch.as_tensor(prior_episode.timesteps, device=device).unsqueeze(0).repeat(venv.num_envs, 1)
        timesteps = torch.cat([prior_timesteps, timesteps], dim=-1)
        init_idx += len(prior_episode)

    states_init = venv.reset()
    states[:, init_idx] = torch.as_tensor(states_init, device=device)
    returns[:, init_idx] = torch.as_tensor(target_return, device=device)

    terminated, truncated = False, False

    # float64 b/c rewards come back as float64 and faster to not convert every time
    venv_episode_return = torch.zeros(num_envs, dtype=torch.float64)
    venv_episode_len = torch.zeros(num_envs, dtype=torch.int)
    dones = torch.zeros(num_envs, dtype=torch.bool)

    # output sequential when actions are needed sequentially, i.e. for autoregressive models with single action output
    def _output_sequential(step, states, actions, returns, timesteps, dones):
        for act_i in range(env_spec.action_dim):
            act_i_output = model(
                states=states[~dones, : step + 1][:, -seq_len:],
                actions=actions[~dones, : step + 1][:, -seq_len:],
                returns_to_go=returns[~dones, : step + 1][:, -seq_len:],
                timesteps=timesteps[~dones, : step + 1][:, -seq_len:],
            )

            # output logits will increase until we hit seq len, so just take last values along dim 1
            actions[:, step, act_i] = act_i_output.logits[:, -1, act_i]
        return actions[:, : step + 1, :]

    for step in trange(max_episode_len, desc="Eval Rollout"):
        if output_sequential:
            logits = _output_sequential(step, states, actions, returns, timesteps, dones)
        else:
            logits = model(
                states=states[~dones, : step + 1][:, -seq_len:],
                actions=actions[~dones, : step + 1][:, -seq_len:],
                returns_to_go=returns[~dones, : step + 1][:, -seq_len:],
                timesteps=timesteps[~dones, : step + 1][:, -seq_len:],
            ).logits

        predicted_action = logits.reshape(num_envs, -1)
        predicted_action = predicted_action[~dones, -env_spec.action_dim :].squeeze()

        # unpack
        next_state, rew, *step_dones, info = venv.step(predicted_action.cpu().numpy())

        if len(step_dones) == 2:
            terminated, truncated = step_dones
            step_dones = terminated | truncated
        else:
            step_dones = step_dones[0]

        dones[step_dones] = True

        next_state = fix_obs_dict(next_state)
        # convert np to torch otherwise ~dones will not work on np arrays when 1 env, e.g. you get
        # `index 1 is out of bounds for axis 0 with size 1`
        next_state = torch.as_tensor(next_state, device=device, dtype=torch.float)
        rew = torch.as_tensor(rew, dtype=torch.float, device=device)

        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[~dones, step] = predicted_action.view(num_envs, -1)[~dones]
        states[~dones, step + 1] = next_state[~dones]
        returns[~dones, step + 1] = returns[~dones, step] - rew[~dones]
        # breakpoint()
        # actions[~dones, step] = torch.as_tensor(predicted_action[~dones], device=device)
        # states[~dones, step + 1] = torch.as_tensor(next_state[~dones], device=device, dtype=torch.float)
        # returns[~dones, step + 1] = returns[~dones, step] - torch.as_tensor(rew[~dones], dtype=torch.float).to(device)

        venv_episode_return[~dones] += rew[~dones].to(venv_episode_return.device)
        venv_episode_len += ~dones  # or (~dones).to(int)

        if (dones == True).all():
            venv_episode_len += 1  # account for last step
            break

    return venv_episode_return, venv_episode_len
