import wandb

def make_wandb_run(project, name=None, config=None):
    if project is not None:
        wandb.init(
            project=project,
            name=name,
            config=config)


