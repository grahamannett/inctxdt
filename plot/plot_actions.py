import matplotlib.pyplot as plt
import numpy as np

from inctxdt.run import make_dataset_from_config, Config
import pyrallis


def plot_histogram(dataset, dataset_name, max_actions: int = 6):
    actions = np.concatenate([v["actions"] for v in dataset.dataset])

    num_actions = min(max_actions, actions.shape[-1])

    n_rows, n_cols = 2, num_actions // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    fig.suptitle(f"Distribution of Actions Before Tokenization for {dataset_name}", fontsize=20)

    axs = axs.ravel()  # Flatten the axis array for easier indexing
    for i in range(len(axs)):
        axs[i].hist(actions[:, i], bins="auto", edgecolor="k", alpha=0.7)
        axs[i].set_title(f"Values of Action {i+1}")
        axs[i].set_xlabel("Count")
        axs[i].set_ylabel("Frequency")

    plt.tight_layout()
    fig.savefig(f"output/plots/histogram_{dataset_name}.png")


@pyrallis.wrap()
def main(config: Config):
    dataset_name = config.env_name

    if isinstance(dataset_name, list):
        for name in dataset_name:
            dataset = make_dataset_from_config(config, name)
            plot_histogram(dataset, name)
    else:
        dataset = make_dataset_from_config(config, dataset_name)
        plot_histogram(dataset, dataset_name)


if __name__ == "__main__":
    main()
