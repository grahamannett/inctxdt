import wandb
import torch
from run_info import antmaze_umaze_v2, wandb_base_path, plot_data_output_dir

run_collection = "antmaze_umaze_v2"

data = {}

api = wandb.Api()
for run, run_id in antmaze_umaze_v2.items():
    df = api.run(f"{wandb_base_path}/{run_id}").history(samples=100_000)
    data[run] = df

torch.save(data, f"{plot_data_output_dir}/{run_collection}.pt")
