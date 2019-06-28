import argparse

import torch
import torchvision
from torchvision.transforms import transforms

import hyperparameter_presets
from experiment_base import ExperimentRunner, ShuffleNetExperimentRunner
from networks import FullyConnectedMNIST, ShuffleNet
from utils import apply_mask_dict_to_weight_dict


def mnist_experiment():
    num_classes = hyperparameter_presets.FULLY_CONNECTED_MNIST['num_classes']
    input_size = hyperparameter_presets.FULLY_CONNECTED_MNIST['input_size']
    hidden_sizes = hyperparameter_presets.FULLY_CONNECTED_MNIST['hidden_sizes']
    batch_size = hyperparameter_presets.FULLY_CONNECTED_MNIST['batch_size']
    learning_rate = hyperparameter_presets.FULLY_CONNECTED_MNIST['learning_rate']
    num_epochs = hyperparameter_presets.FULLY_CONNECTED_MNIST['epochs']
    prune_percent = hyperparameter_presets.FULLY_CONNECTED_MNIST['prune_percent']
    pruning_iterations = hyperparameter_presets.FULLY_CONNECTED_MNIST['prune_iterations']

    # Temporary parameters. Should probably move this to the hyper parameters file as well
    num_training = 55000
    num_validation = 5000

    # Prepare the dataset
    to_tensor_transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = torchvision.datasets.MNIST(root='datasets/', train=True, download=True, transform=to_tensor_transform)
    mnist_test_dataset = torchvision.datasets.MNIST(root='datasets/', train=False, transform=to_tensor_transform)
    mask = list(range(num_training))
    mnist_train_dataset = torch.utils.data.Subset(mnist_dataset, mask)
    mask = list(range(num_training, num_training + num_validation))
    mnist_val_dataset = torch.utils.data.Subset(mnist_dataset, mask)

    # Load the dataset
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)
    mnist_val_loader = torch.utils.data.DataLoader(dataset=mnist_val_dataset, batch_size=batch_size, shuffle=False)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=False)

    model = FullyConnectedMNIST(input_size, hidden_sizes, num_classes)
    if torch.cuda.is_available():
        model.cuda()

    experiment = ExperimentRunner(model, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)

    experiment.train(input_size, mnist_train_loader, mnist_val_loader)
    experiment.test(input_size, mnist_test_loader)
    mask_dict = experiment.get_initial_mask()

    # I now have a mask and pre_init
    # 1. Initialize another network with pre_init, and set a mask on it
    # 2. while training, make sure that the weights are indeed 0 according to the mask - set a breakpoint
    # after an epoch and verify this
    # 3. Repeat 1 & 2 one more time
    # 4. Check the test accuracy of the pruned network on the test data set.

    mask_dict = experiment.prune(mask_dict, prune_percent=prune_percent)
    # experiment.print_stats()

    for i in range(1, pruning_iterations):
        initial_weights_after_mask = apply_mask_dict_to_weight_dict(mask_dict, experiment.model.initial_weights)
        new_model = FullyConnectedMNIST(input_size, hidden_sizes, num_classes, pre_init=initial_weights_after_mask, mask_dict=mask_dict)
        if torch.cuda.is_available():
            new_model.cuda()
        experiment.set_model(new_model)
        experiment.train(input_size, mnist_train_loader, mnist_val_loader)
        experiment.test(input_size, mnist_test_loader)
        try:
            mask_dict = experiment.prune(mask_dict, prune_percent=prune_percent)
        except IndexError:
            # Can happen when we try to prune too much
            break

    experiment.plot()
    experiment.print_stats()


def shufflenet_experiment():
    num_classes = hyperparameter_presets.SHUFFLENET['num_classes']
    input_size = hyperparameter_presets.SHUFFLENET['input_size']
    hidden_sizes = hyperparameter_presets.SHUFFLENET['hidden_sizes']
    batch_size = hyperparameter_presets.SHUFFLENET['batch_size']
    learning_rate = hyperparameter_presets.SHUFFLENET['learning_rate']
    num_epochs = hyperparameter_presets.SHUFFLENET['epochs']
    prune_percent = hyperparameter_presets.SHUFFLENET['prune_percent']
    pruning_iterations = hyperparameter_presets.SHUFFLENET['prune_iterations']

    # Temporary parameters. Should probably move this to the hyper parameters file as well
    num_training = 49000
    num_validation = 1000

    norm_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])
    cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                                 train=True,
                                                 transform=norm_transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                                train=False,
                                                transform=test_transform
                                                )

    mask = list(range(num_training))
    train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
    mask = list(range(num_training, num_training + num_validation))
    val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    model = ShuffleNet(input_size, num_classes)
    if torch.cuda.is_available():
        model.cuda()

    experiment = ShuffleNetExperimentRunner(model, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)

    experiment.train(input_size, train_loader, val_loader)
    experiment.print_stats()
    # experiment.test(input_size, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run lottery ticket experiments')
    parser.add_argument('--experiment', help='Name of the experiment')

    args = parser.parse_args()
    if args.experiment == 'mnist':
        mnist_experiment()
    elif args.experiment == 'shufflenet':
        shufflenet_experiment()
    else:
        print("Invalid value for 'experiment'")
