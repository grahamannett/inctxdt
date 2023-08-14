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


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
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

        output = model(  # fix this noqa!!!
            states=states[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            actions=actions[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            returns_to_go=returns[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            timesteps=timesteps[:, : step + 1][:, -env_spec.seq_len :],  # noqa
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
    venv: gymnasium.vector.VectorEnv,
    env_spec: EnvSpec,
    device: str,
    target_return: float,
    output_sequential: bool = False,
    prior_episode: EpisodeData = None,
) -> Tuple[float, float]:
    assert isinstance(venv, (gymnasium.vector.VectorEnv, VectorEnv)), "venv must be a vectorized env."

    num_envs = venv.num_envs
    max_episode_len = episode_len = env_spec.episode_len
    seq_len = env_spec.seq_len
    model.eval()

    if prior_episode is not None:
        episode_len += len(prior_episode)

    states = torch.zeros(num_envs, episode_len + 1, env_spec.state_dim, dtype=torch.float, device=device)
    actions = torch.zeros(num_envs, episode_len, env_spec.action_dim, dtype=torch.float, device=device)
    returns = torch.zeros(num_envs, episode_len + 1, dtype=torch.float, device=device)
    timesteps = torch.arange(env_spec.episode_len, dtype=torch.long, device=device)
    timesteps = timesteps.repeat(num_envs, 1).view(num_envs, -1)

    states_init = venv.reset()

    if len(states_init) == 2:
        states_init = states_init[0]

    # _check_states(states_init)  # # states_init = fix_obs_dict(states_init)

    init_idx = 0
    if prior_episode:
        states[:, : len(prior_episode)] = torch.as_tensor(prior_episode.states, device=device)
        actions[:, : len(prior_episode)] = torch.as_tensor(prior_episode.actions, device=device)
        returns[:, : len(prior_episode)] = torch.as_tensor(prior_episode.returns_to_go, device=device)
        prior_ts = torch.as_tensor(prior_episode.timesteps, device=device).unsqueeze(0).repeat(venv.num_envs, 1)
        timesteps = torch.cat([prior_ts, timesteps], dim=-1)
        init_idx = len(prior_episode)

    states[:, init_idx] = torch.as_tensor(states_init, device=device)
    returns[:, init_idx] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    terminated, truncated = False, False

    # float64 b/c rewards come back as float64 and faster to not convert every time
    venv_episode_return = torch.zeros(num_envs, dtype=torch.float64)
    venv_episode_len = torch.zeros(num_envs, dtype=torch.int)
    dones = torch.zeros(num_envs, dtype=torch.bool)

    for step in trange(max_episode_len, desc="Eval Rollout"):
        step += init_idx
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)

        output = model(
            states[~dones, : step + 1][:, -seq_len:],  # noqa
            actions[~dones, : step + 1][:, -seq_len:],  # noqa
            returns[~dones, : step + 1][:, -seq_len:],  # noqa
            timesteps[~dones, : step + 1][:, -seq_len:],  # noqa
        )
        logits = output.logits

        if output_sequential:
            actions_ = [output.logits]

            for i in range(1, env_spec.action_dim):
                output = model(
                    states[~dones, : step + 1][:, -seq_len:],  # noqa
                    actions[~dones, : step + 1][:, -seq_len:],  # noqa
                    returns[~dones, : step + 1][:, -seq_len:],  # noqa
                    timesteps[~dones, : step + 1][:, -seq_len:],  # noqa
                )
                actions_.append(output.logits)

            logits = torch.stack(actions_, dim=-1)

        predicted_action = logits.reshape(venv.num_envs, -1)
        predicted_action = predicted_action[~dones, -env_spec.action_dim :].squeeze().cpu().numpy()

        # unpack
        next_state, rew, *step_dones, info = venv.step(predicted_action)

        if len(step_dones) == 2:
            terminated, truncated = step_dones
            step_dones = terminated | truncated
        else:
            step_dones = step_dones[0]

        # NOTE: there will be truncated and terminated later - throw error and catch this
        # assert len(step_dones) == 1, "not sure if i handle these correctly. need to handle terminated/truncated"

        dones[step_dones] = True

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
    env = gymnasium.vector.make("halfcheetah-medium-v2", num_envs=10, asynchronous=False)

    vals = venv_eval_rollout(model, env, env_spec, "cpu", 12000.0)
    print(vals)
