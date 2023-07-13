import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import minari
from minari import DataCollectorV0


torch.manual_seed(42)


env = DataCollectorV0(gym.make('CartPole-v1'))
path = os.path.abspath('') + '/logs/ppo/CartPole-v1_1/best_model'
# path = "logs/ppo/CartPole-v1_1/best_model"
agent = PPO.load(path)

total_episodes = 1_000
for i in tqdm(range(total_episodes)):
    obs, _ = env.reset(seed=42)
    while True:
        action, _ = agent.predict(obs)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = minari.create_dataset_from_collector_env(dataset_id="CartPole-v1-expert-local-ppo",
                                                   collector_env=env,
                                                   algorithm_name="ExpertPolicy",
                                                   author="Graham",
                                                   )

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "seed": torch.Tensor([x.seed for x in batch]),
        "total_timesteps": torch.Tensor([x.total_timesteps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch],
            batch_first=True
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )
    }

minari_dataset = minari.load_dataset("CartPole-v1-expert-local-ppo")
dataloader = DataLoader(minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

env = minari_dataset.recover_environment()
observation_space = env.observation_space
action_space = env.action_space
assert isinstance(observation_space, spaces.Box)
assert isinstance(action_space, spaces.Discrete)

policy_net = PolicyNetwork(np.prod(observation_space.shape), action_space.n)
optimizer = torch.optim.Adam(policy_net.parameters())
loss_fn = nn.CrossEntropyLoss()


num_epochs = 32

for epoch in range(num_epochs):
    for batch in dataloader:
        a_pred = policy_net(batch['observations'][:, :-1])
        a_hat = F.one_hot(batch["actions"]).type(torch.float32)
        loss = loss_fn(a_pred, a_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=42)
done = False
accumulated_rew = 0
while not done:
    action = policy_net(torch.Tensor(obs)).argmax()
    obs, rew, ter, tru, _ = env.step(action.numpy())
    done = ter or tru
    accumulated_rew += rew

env.close()
print("Accumulated rew: ", accumulated_rew)