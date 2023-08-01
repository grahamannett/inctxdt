from torch.utils.data import DataLoader

from inctxdt.batch import Collate
from inctxdt.datasets import AcrossEpisodeDataset, MultipleMinariDataset, MinariDataset
from inctxdt.d4rl_datasets import D4rlDataset
from inctxdt.model import DecisionTransformer
from inctxdt.trainer import config, train

from accelerate import Accelerator

# robot envs are envs from mujoco that have control tasks/vectors for actions
dataset_names = ["halfcheetah-medium-v2", ""]


def main():
    pass


if __name__ == "__main__":
    main()
