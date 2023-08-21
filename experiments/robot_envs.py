from torch.utils.data import DataLoader

from inctxdt.batch import Collate
from inctxdt.minari_datasets import AcrossEpisodeDataset, MultipleMinariDataset, MinariDataset
from inctxdt.d4rl_datasets import D4rlDataset
from inctxdt.models.model import DecisionTransformer
from inctxdt.trainer import config, train

from accelerate import Accelerator

reward_scale: float = 0.001
target_return: float = 12000.0

# robot envs are envs from mujoco that have control tasks/vectors for actions
dataset_names = ["halfcheetah-medium-v2", ""]
dataset_names = [
        "halfcheetah-random-v2",
        "halfcheetah-medium-v2",
        "halfcheetah-expert-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-full-replay-v2",
        "halfcheetah-medium-expert-v2",
    ]

    datasets = [
        D4rlAcrossEpisodeDataset(dataset_name=n, seq_len=config.seq_len, reward_scale=config.reward_scale)
        for n in dataset_names
    ]

def main():
    pass


if __name__ == "__main__":
    main()
