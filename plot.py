import wandb
api = wandb.Api()
run = api.run("/graham/inctxdt/runs/5e4hng9l")
returns_std = run.history(pandas=(True), keys=["returns_std",])

