from torch.utils.data import DataLoader
from inctxdt.model import DecisionTransformer


from inctxdt.trainer import train, config
from inctxdt.batch import Collate
from inctxdt.datasets import AcrossEpisodeDataset

config.get()

ds = AcrossEpisodeDataset(env_name="pointmaze-medium-v1", max_num_epsisodes=3)
dataloader = DataLoader(
    ds,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=Collate(device=config.device, batch_first=True),
)

env = ds.recover_environment()

sample = ds[0]

state_dim = sample.observations.shape[-1]
action_dim = sample.actions.shape[-1]

model = DecisionTransformer(state_dim=state_dim, action_dim=action_dim, embedding_dim=128, num_layers=6)

train(model, dataloader=dataloader, config=config)
