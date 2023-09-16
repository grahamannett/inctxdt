import pyrallis


def main():
    config = pyrallis.parse(config_class=Config)
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    # breakpoint()
    dataset = make_dataset_from_config(config)

    _, env, venv, obs_space, act_space = get_env(config=config, dataset=dataset)

    env_spec = EnvSpec(
        episode_len=config.episode_len,
        seq_len=config.seq_len,
        env_name=getattr(dataset, "env_name", dataset.dataset_name),
        action_dim=act_space.shape[0],
        state_dim=obs_space.shape[0],
    )

    dataset.state_dim = env_spec.state_dim
    dataset.action_dim = env_spec.action_dim

    # usually i try to init tracker last so i have last few seconds to exit script if needed
    accelerator = Accelerator(log_with="wandb")
    init_trackers(accelerator, config)

    accelerator.print(config)
    dispatch_cmd[config.cmd](config, dataset=dataset, accelerator=accelerator, env_spec=env_spec, env=env, venv=venv)


if __name__ == "__main__":
    main()
