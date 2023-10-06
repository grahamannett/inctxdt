# pt_dfs = torch.load(f"{plot_data_output_dir}/antmaze_umaze_v2.pt")
# df = pt_dfs["baseline"]

# to_plot = df[["_step", "eval/0.5_return_mean"]].dropna()

# plt.figure(figsize=(10, 6))
# plt.plot(to_plot["_step"], to_plot["eval/0.5_return_mean"], label="baseline")

# plt.fill_between(
#     df["_step"],
#     (df["eval/0.5_return_mean"] - df["eval/0.5_return_std"]),
#     (df["eval/0.5_return_mean"] + df["eval/0.5_return_std"]),
#     color="blue",
#     alpha=0.1,
# )


# plt.tight_layout()
# plt.savefig("plot.png")


import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import numpy as np

from scipy.interpolate import make_interp_spline


from run_info import plot_data_output_dir, run_collections

sns.set_theme()

group = "walker2d-medium_expert_v2"
data_file = f"output/plot_data/{group}.pt"

cutoff = 100
n_spline = 1000
columns = ["eval/2500.0_return"]

job_types: Dict[str, List[pd.DataFrame]] = torch.load(data_file)

print("job-types:", *list(job_types.keys()))


def smooth_line(col_data, col_steps):
    idx = range(len(col_data))
    xnew = np.linspace(min(idx), max(idx), n_spline)

    # interpolation
    spl = make_interp_spline(idx, col_data, k=3)
    smooth = spl(xnew)

    xnew = np.linspace(min(col_steps), max(col_steps), n_spline)
    return xnew, smooth


def get_plot_data(job_type, column) -> Tuple[List[float], List[float]]:
    data_means = []
    data_stds = []

    mean_column_name = f"{column}_mean"
    std_column_name = f"{column}_std"

    mean_col_lengths = []
    std_col_lengths = []

    # print(job_types[job_type][0].columns)

    for full_df in job_types[job_type]:
        df = full_df[["_step", mean_column_name, std_column_name]].dropna()

        if len(df) < cutoff:
            continue

        mean_column = df[mean_column_name].values
        std_column = df[std_column_name].values

        mean_column = mean_column.clip(min=0)

        steps = df["_step"].values

        mean_col_lengths.append(len(mean_column))
        std_col_lengths.append(len(std_column))

        data_means.append(mean_column)
        data_stds.append(std_column)

    # print(mean_col_lengths, std_col_lengths)

    return job_type, column, np.stack(data_means), np.stack(data_stds), steps, len(job_types[job_type])


rename_job_type = {
    "baseline": "Baseline",
    "tokenized": "Tokenized Actions",
    "tokenized_spread": "Tokenized Actions (Unpacked Into Time)",
    "tokenized_seperate": "Tokenized Actions (Unpacked Into Time)",
}


def plot_column(ax, job_type, column, data_mean, data_std, steps, n_runs):
    col_mean = data_mean.mean(axis=0)
    col_std = data_mean.std(axis=0)

    col_std_mean = data_std.mean(axis=0)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    label = rename_job_type.get(job_type, job_type)
    ax.plot(steps, col_mean, label=label, linewidth=5, alpha=0.8)
    ax.fill_between(steps, col_mean - col_std_mean, col_mean + col_std_mean, alpha=0.1)


fig, axs = plt.subplots(1, 1, figsize=(10, 6))


# col_data, col_std_data, col_steps, n_runs = get_plot_data("baseline", "eval/5000.0_return")
# plot_column(axs, "baseline", "eval/5000.0_return", col_data, col_std_data, col_steps, n_runs)
plot_column(axs, *get_plot_data("baseline", "eval/5000.0_return"))
plot_column(axs, *get_plot_data("tokenized", "eval/5000.0_return"))
# plot_column(axs, *get_plot_data("tokenized_spread", "eval/5000.0_return"))
plot_column(axs, *get_plot_data("tokenized_seperate", "eval/5000.0_return"))


axs.legend()
axs.set_xlim([-500, 25_000])
fig.suptitle(f"Returns for {group}", fontsize=24)
fig.set_tight_layout(True)
fig.savefig("output/plots/returns-walker2d.png")
