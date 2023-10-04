import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from run_info import plot_data_output_dir, run_collections

sns.set_theme()

pt_dfs = torch.load(f"{plot_data_output_dir}/antmaze_umaze_v2.pt")
df = pt_dfs["baseline"]

to_plot = df[["_step", "eval/0.5_return_mean"]].dropna()

plt.figure(figsize=(10, 6))
plt.plot(to_plot["_step"], to_plot["eval/0.5_return_mean"], label="baseline")

plt.fill_between(
    df["_step"],
    (df["eval/0.5_return_mean"] - df["eval/0.5_return_std"]),
    (df["eval/0.5_return_mean"] + df["eval/0.5_return_std"]),
    color="blue",
    alpha=0.1,
)


plt.tight_layout()
plt.savefig("plot.png")
