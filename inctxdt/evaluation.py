from typing import Dict, NamedTuple, Tuple

import numpy as np
import torch
from gym import Env

from inctxdt.config import config_tool

# from gymnasium import Env
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
    env: Env,
    config: config_tool,
    device: str,
    target_return: float,
) -> Tuple[float, float]:
    model.eval()
    env_spec = config.get_env_spec(env)
    states = torch.zeros(
        1,
        env_spec.episode_len + 1,
        env_spec.state_dim,
        dtype=torch.float,
        device=device,
    )
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
