import pandas as pd
import wandb
import torch

api = wandb.Api()

# Project is specified by <entity/project-name>

group = "walker2d-medium_expert_v2"
outfile = f"output/plot_data/{group}.pt"
runs = api.runs("graham/inctxdt", filters={"group": group})

summary_list, config_list, name_list = [], [], []

job_types = {}
for run in runs:
    data = run.history(samples=100000)

    if run.job_type not in job_types:
        job_types[run.job_type] = []

    job_types[run.job_type].append(data)


torch.save(job_types, outfile)
