# A file which stores common hyper parameter presents for convenience.

FULLY_CONNECTED_MNIST = {
    'input_size': 28 * 28 * 1,
    'hidden_sizes': [300, 100],
    'num_classes': 10,
    'batch_size': 100,
    'learning_rate': 0.0012,
    'epochs': 100,
    'prune_percent': 0.2,
    'prune_iterations': 40
}

SHUFFLENET = {
    'input_size': 32 * 32 * 3,
    'num_classes': 10,
    'batch_size': 100,
    'learning_rate': 0.01,
    'epochs': 30,
    'prune_percent': 0.1,
    'prune_iterations': 40,
    'reg': 0,
    'lr_step_size': 10,
    'lr_step_gamma': 0.1,
    'decay_lr': True
}

CONV2 = {
    'input_size': 32 * 32 * 3,
    'num_classes': 10,
    'batch_size': 100,
    'learning_rate': 0.0002,
    'epochs': 15,
    'prune_percent': 0.2,
    'prune_iterations': 35
}
