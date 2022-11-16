import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # optimizer
    config.lr = 3e-4
    config.b1 = 0.9
    config.b2 = 0.999
    config.bs = 32
    config.train_steps = 10_000

    # dataloader
    config.workers = 0

    config.logging_interval = 10

    return config
