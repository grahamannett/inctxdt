import pyrallis
import torch
from accelerate import Accelerator
from inctxdt.batch import Collate

from inctxdt.config import Config, EnvSpec
from inctxdt.d4rl_datasets import D4rlAcrossEpisodeDataset, D4rlDataset
from inctxdt.evaluation import venv_eval_rollout
from inctxdt.env_helper import get_env
from inctxdt.models.model import DecisionTransformer

from inctxdt.trainer import train_embeds


def eval_inctx(model, accelerator, config):
    pass


def eval_baseline(model, accelerator, config):
    _, env, venv, obs_space, action_space = get_env(config)

    env_spec = EnvSpec(
        episode_len=config.episode_len,
        seq_len=config.seq_len,
        env_name=config.env_name,
        state_dim=obs_space.shape[0],
        action_dim=action_space.shape[0],
    )

    eval_ret, _ = venv_eval_rollout(
        model,
        venv,
        env_spec,
        target_return=config.target_returns[0] * config.reward_scale,
        device=accelerator.device,
    )

    print(f"eval_returns:{(eval_ret / config.reward_scale).mean().item()}")


config = pyrallis.parse(config_class=Config)
torch.cuda.manual_seed(config.seed)
torch.manual_seed(config.seed)
_, env, venv, obs_space, action_space = get_env(config)

accelerator = Accelerator()
# model = torch.load(f"{config.exp_dir}/model_base")
# dataset = D4rlDataset(dataset_name=config.dataset_name, reward_scale=config.reward_scale)
dataset = D4rlAcrossEpisodeDataset(dataset_name=config.env_name, seq_len=100, reward_scale=config.reward_scale)
env_spec = EnvSpec(
    episode_len=config.episode_len,
    seq_len=config.seq_len,
    env_name=config.env_name,
    state_dim=obs_space.shape[0],
    action_dim=action_space.shape[0],
)


model = DecisionTransformer(
    state_dim=env_spec.state_dim,
    action_dim=env_spec.action_dim,
    embedding_dim=128,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    seq_len=config.seq_len,
    env_spec=env_spec,
)
model.load_state_dict(torch.load(f"{config.exp_dir}/model_3/pytorch_model.bin"))

model.to(accelerator.device)


episode = None
episode = dataset[0]

layers_to_train = [
    "embed_output_layers.state_emb",
]
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=config.shuffle,
    num_workers=config.num_workers,
    collate_fn=Collate(
        batch_first=True,
        device=None if accelerator else config.device,
        return_fn=dataset.collate_fn(batch_first=True),
    ),
)

# model = train_embeds(config, model, layers_to_train, dataloader=dataloader, accelerator=accelerator)
eval_ret, _ = venv_eval_rollout(
    model,
    venv,
    env_spec,
    target_return=config.target_return * config.reward_scale,
    device=accelerator.device,
    # prior_episode=episode,
)

print(f"eval_returns:{(eval_ret / config.reward_scale).mean().item()}")
