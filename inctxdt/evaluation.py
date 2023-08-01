from typing import Dict, NamedTuple, Tuple

import numpy as np
import torch
from gymnasium import Env
from inctxdt.model import DecisionTransformer


def flatten_obs_dict(obs_dict: Dict[str, np.array]) -> np.array:
    return np.concatenate([obs_dict[key] for key in obs_dict.keys()], axis=-1)


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: Env,
    env_spec: NamedTuple,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1,
        env_spec.episode_len + 1,
        env_spec.state_dim,
        dtype=torch.float,
        device=device,
    )
    actions = torch.zeros(
        1, env_spec.episode_len, env_spec.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, env_spec.episode_len + 1, dtype=torch.float, device=device)

    time_steps = torch.arange(env_spec.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    if isinstance(states_init := env.reset()[0], dict):
        states_init = flatten_obs_dict(states_init)
        # states_init = states_init["observation"]

    states[:, 0] = torch.as_tensor(states_init, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    terminated, truncated = False, False
    episode_return, episode_len = 0.0, 0.0
    for step in range(env_spec.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            actions[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            returns[:, : step + 1][:, -env_spec.seq_len :],  # noqa
            time_steps[:, : step + 1][:, -env_spec.seq_len :],  # noqa
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        breakpoint()

        # unpack
        next_state, reward, *term_trunc, info = env.step(predicted_action)
        done = term_trunc if len(term_trunc) == 1 else (term_trunc[0] or term_trunc[1])
        # terminated, truncated = term_trunc if len(other) > 1 else (other, other)

        if isinstance(next_state, dict):
            next_state = flatten_obs_dict(next_state)
            # next_state = next_state["observation"]

        done = terminated or truncated
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
