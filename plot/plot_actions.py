import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import seaborn as sns
import torch

from inctxdt.config import Config, PlotConfig
from inctxdt.run import make_dataset_from_config

# change to seaborn

# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.rcParams["text.usetex"] = True

# sns.set_theme()
# sns.set_style("dark")

plt.style.use("ggplot")


def plot_histogram(dataset, dataset_name, max_actions: int = 6, plot_config: PlotConfig = None):
    actions = np.concatenate([v["actions"] for v in dataset.dataset])

    torch.save(
        {
            "dataset_name": dataset_name,
            "actions": actions,
        },
        f"output/histogram/actions-{dataset_name}.pt",
    )
    # np.save(f"output/histogram/actions-{dataset_name}.npy", actions)

    num_actions = min(max_actions, actions.shape[-1])

    n_rows = plot_config.n_rows
    n_cols = plot_config.n_columns if not plot_config.plot_all_actions else num_actions // n_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=plot_config.figsize)

    plot_title = plot_config.plot_title or f"Distribution of Actions Before Tokenization for {dataset_name}"
    fig.suptitle(plot_title, fontsize=plot_config.plot_title_fontsize)

    axs = axs.ravel()  # Flatten the axis array for easier indexing
    for i in range(len(axs)):
        axs[i].hist(actions[:, i], bins="auto", edgecolor="k", color="tab:blue", alpha=0.7)

        # set initial and then change if we pass
        subplot_title = f"Action {i+1}"
        if plot_config.use_subplot_titles and len(plot_config.subplot_titles) > i:
            subplot_title = rf"{plot_config.subplot_titles[i]}"

        axs[i].set_title(subplot_title, fontsize=plot_config.subplot_title_fontsize)
        axs[i].set_xlabel("Count", fontsize=plot_config.subplot_xlabel_fontsize)
        axs[i].set_ylabel("Frequency", fontsize=plot_config.subplot_ylabel_fontsize)

    plt.tight_layout()
    fig.savefig(f"output/plots/{plot_config.plot_name}.png")  # was {plot_config.plot_name}_{dataset_name}.png


@pyrallis.wrap()
def main(config: Config):
    dataset_name = config.env_name

    if isinstance(dataset_name, list):
        for name in dataset_name:
            dataset = make_dataset_from_config(config, name)
            plot_histogram(dataset, name, plot_config=config.plot)
    else:
        dataset = make_dataset_from_config(config, dataset_name)
        plot_histogram(dataset, dataset_name, plot_config=config.plot)


if __name__ == "__main__":
    main()
