import time
import torch
import torchvision

from tqdm import tqdm
from torch import nn
from torchvision.transforms import transforms

import hyperparameter_presets
import matplotlib.pyplot as plt

# A helper class that takes a model and dataset, and runs the experiment on it.
from networks import FullyConnectedMNIST
from utils import get_zero_count, apply_mask_dict_to_weight_dict


class ExperimentRunner:
    TRAINING_DURATION_SECONDS = "training_duration_seconds"
    FINAL_VALIDATION_ACCURACY = "final_validation_accuracy"
    TEST_ACCURACY = "test_accuracy"
    BEST_VALIDATION_ACCURACY = " best_validation_accuracy"
    DEVICE = "device"
    ZERO_PERCENTAGE_IN_INITIAL_WEIGHTS = "zero_percentage_in_initial_weights"
    ZERO_PERCENTAGE_IN_MASKS = "zero_percentage_in_masks"
    PERCENTAGE_WEIGHT_MASKED = "percentage_weight_masked"

    def __init__(self, model, num_epochs=10, batch_size=200, learning_rate=5e-3, learning_rate_decay=0.95):
        self.model = model
        self.learning_rate = learning_rate
        self.reg = 0.001  # Should this be a hyper parameter?
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate_decay = learning_rate_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = {
            self.DEVICE: str(self.device)
        }
        self.update_stat(self.ZERO_PERCENTAGE_IN_INITIAL_WEIGHTS, self.get_zero_count_in_weights())
        self.update_stat(self.PERCENTAGE_WEIGHT_MASKED, self.model.get_percent_weights_masked())

    def print_stats(self):
        print(self.stats)

    def update_stat(self, stat_name, value):
        self.stats[stat_name] = value

    @staticmethod
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, input_size, train_dataloader, validation_dataloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg)

        training_start_time = time.time()
        best_validation_accuracy_so_far = 0
        for epoch in tqdm(range(self.num_epochs)):
            for i, (images, labels) in enumerate(train_dataloader):
                # Move tensors to the configured device
                images = images.to(self.device)
                labels = labels.to(self.device)
                images = images.view(self.batch_size, input_size)
                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

            # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))
            # lr = self.learning_rate * self.learning_rate_decay
            # self.update_lr(optimizer, lr)
            validation_accuracy = self.validate(input_size, validation_dataloader)
            if validation_accuracy > best_validation_accuracy_so_far:
                best_validation_accuracy_so_far = validation_accuracy
                self.update_stat(self.BEST_VALIDATION_ACCURACY, best_validation_accuracy_so_far)
                torch.save(self.model.state_dict(), 'temp.ckpt')

        self.update_stat(self.TRAINING_DURATION_SECONDS, time.time() - training_start_time)
        self.update_stat(self.FINAL_VALIDATION_ACCURACY, validation_accuracy)

    def validate(self, input_size, validation_dataloader):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validation_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                images = images.view(self.batch_size, input_size)
                scores = self.model.forward(images)

                predicted = []

                def get_class(x):
                    return torch.argsort(x)[-1]

                for i in range(0, len(scores)):
                    predicted.append(get_class(scores[i]))

                predicted = torch.stack(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            validation_accuracy = 100 * correct / total
            # print('Validation accuracy is: {} %'.format(validation_accuracy))

        return validation_accuracy

    def test(self, input_size, test_dataloader):
        best_model = FullyConnectedMNIST(self.model.input_size, self.model.hidden_sizes, self.model.num_classes)
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load('temp.ckpt'))

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                images = images.view(self.batch_size, input_size)
                scores = best_model.forward(images)

                predicted = []

                def get_class(x):
                    return torch.argsort(x)[-1]

                for i in range(0, len(scores)):
                    predicted.append(get_class(scores[i]))

                predicted = torch.stack(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_accuracy = 100 * correct / total
            print('Test accuracy is: {} %'.format(test_accuracy))
            print('Best validation accuracy is: {} %'.format(self.stats[self.BEST_VALIDATION_ACCURACY]))
        self.update_stat(self.TEST_ACCURACY, test_accuracy)
        return test_accuracy

    def get_initial_mask(self):
        mask_dict = dict()
        for name, parameter in self.model.named_parameters():
            if name.endswith('weight'):
                mask_dict[name] = torch.ones(parameter.data.shape)

        return mask_dict

    def prune(self, mask_dict, prune_percent=0.1):
        # Use the best model obtained through early stopping. Weights are in the file temp.ckpt
        # TODO: Make this more elegant - do not hardcode the file name
        best_model = FullyConnectedMNIST(self.model.input_size, self.model.hidden_sizes, self.model.num_classes)
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load('temp.ckpt'))

        # We assume that all layers are pruned by the same percentage
        # Yes, we prune per layer, not globally

        for name, parameter in best_model.named_parameters():
            # TODO: Check if we should indeed ignore the bias
            if name.endswith('weight'):
                current_mask = mask_dict.get(name, None)
                if name == 'output_layer.weight':
                    # Last layer always has a different prune rate
                    # TODO: Since this is model specific, move the prune() method to the neural network class
                    new_mask = self.get_new_mask(prune_percent/2, parameter.data, current_mask)
                else:
                    new_mask = self.get_new_mask(prune_percent, parameter.data, current_mask)
                mask_dict[name] = new_mask

        self.update_stat(self.ZERO_PERCENTAGE_IN_MASKS, self.get_zero_count_in_mask(mask_dict))
        return mask_dict

    @staticmethod
    def get_new_mask(prune_percent, data, current_mask):
        # I hate this if statement as much as you do. Currently there's no easy way to switch between CPU and GPU
        if torch.cuda.is_available():
            sorted_weights = torch.sort(torch.abs(torch.masked_select(data, current_mask.cuda().byte()))).values
            cutoff_index = int(torch.tensor(prune_percent * len(sorted_weights)).round())
            cutoff = sorted_weights[cutoff_index]
            return torch.where(torch.abs(data) <= cutoff, torch.zeros(current_mask.shape, dtype=torch.float).cuda(),
                               current_mask.cuda())
        else:
            sorted_weights = torch.sort(torch.abs(torch.masked_select(data, current_mask.byte()))).values
            cutoff_index = int(torch.tensor(prune_percent * len(sorted_weights)).round())
            cutoff = sorted_weights[cutoff_index]
            return torch.where(torch.abs(data) <= cutoff, torch.zeros(current_mask.shape, dtype=torch.float),
                               current_mask)

    def get_zero_count_in_weights(self):
        # In each linear layer in the network, count the number of zeros. Useful for debugging
        zeros_info_dict = dict()

        for name, param in self.model.named_parameters():
            if name.endswith('weight'):
                zeros_info_dict[name] = get_zero_count(param.data)/(param.data.shape[0] * param.data.shape[1])

        return zeros_info_dict

    def get_zero_count_in_mask(self, mask_dict):
        zeros_info_dict = dict()

        for name, data in mask_dict.items():
            zeros_info_dict[name] = get_zero_count(data)/(data.shape[0] * data.shape[1])

        return zeros_info_dict


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

    validation_accuracies = []
    percent_weight_masked_list = []
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
    validation_accuracies.append(experiment.stats[ExperimentRunner.BEST_VALIDATION_ACCURACY])
    percent_weight_masked_list.append(experiment.stats[ExperimentRunner.PERCENTAGE_WEIGHT_MASKED])

    for i in range(1, pruning_iterations):
        initial_weights_after_mask = apply_mask_dict_to_weight_dict(mask_dict, experiment.model.initial_weights)
        new_model = FullyConnectedMNIST(input_size, hidden_sizes, num_classes, pre_init=initial_weights_after_mask, mask_dict=mask_dict)
        if torch.cuda.is_available():
            new_model.cuda()
        experiment = ExperimentRunner(new_model, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)
        experiment.train(input_size, mnist_train_loader, mnist_val_loader)
        experiment.test(input_size, mnist_test_loader)
        mask_dict = experiment.prune(mask_dict, prune_percent=prune_percent)
        validation_accuracies.append(experiment.stats[ExperimentRunner.BEST_VALIDATION_ACCURACY])
        percent_weight_masked_list.append(experiment.stats[ExperimentRunner.PERCENTAGE_WEIGHT_MASKED])
        # experiment.print_stats()

    # TODO: Refactor so that experiment.plot() can do this
    percent_weights = [percent for percent in percent_weight_masked_list]
    accuracies = [accuracy for accuracy in validation_accuracies]
    plt.plot(percent_weights, accuracies)
    plt.xlabel('percentage of weights pruned')
    plt.ylabel('Early stopping val acc.')
    plt.savefig("graph.png")


if __name__ == "__main__":
    mnist_experiment()

