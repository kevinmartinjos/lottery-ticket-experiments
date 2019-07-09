import time
import torch
import torchvision
import json

from tqdm import tqdm
from torch import nn

import matplotlib.pyplot as plt

# A helper class that takes a model and dataset, and runs the experiment on it.
from networks import FullyConnectedMNIST, ShuffleNet, Conv2Net
from utils import get_zero_count


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
        self.learning_rate = learning_rate
        self.reg = 0.001  # Should this be a hyper parameter?
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate_decay = learning_rate_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = []  # Should be an array of dicts
        self.set_model(model)
        self.model = model  # redundant, I know

    def set_model(self, model):
        # The model associated with the experiment keeps changing as we iteratively prune.
        self.model = model
        self.stats.append(dict())  # For storing the stats related to the new model

        # TODO: Refactor so as to make this work for higher dimensional tensors
        self.update_stat(self.ZERO_PERCENTAGE_IN_INITIAL_WEIGHTS, self.get_zero_count_in_weights())
        self.update_stat(self.PERCENTAGE_WEIGHT_MASKED, self.model.get_percent_weights_masked())

    def get_stat(self, param):
        # Gets the param from the latest entry in the stats array
        return self.stats[-1][param]

    def print_stats(self):
        with open('result.json', 'w') as outfile:
            json.dump(self.stats, outfile)

    def update_stat(self, stat_name, value):
        stat = self.stats[-1]
        stat[stat_name] = value

    @staticmethod
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, input_size, train_dataloader, validation_dataloader):
        # TODO: Must return the best validation accuracy (early stopping_
        # TODO: Must automatically update self.stats without the child class being aware of it
        raise NotImplementedError

    def validate(self, input_size, validation_dataloader):
        # TODO: Must automatically update self.stats without the child class being aware of it
        raise NotImplementedError

    def test(self, input_size, test_dataloader):
        # TODO: Must automatically update self.stats without the child class being aware of it
        raise NotImplementedError

    def prune(self, mask_dict, prune_percent=0.1):
        raise NotImplementedError

    def get_initial_mask(self):
        mask_dict = dict()
        for name, parameter in self.model.named_parameters():
            if name.endswith('weight'):
                mask_dict[name] = torch.ones(parameter.data.shape)

        return mask_dict

    @staticmethod
    def get_new_mask(prune_percent, data, current_mask):
        # Coincidentally, this works tensors of any dimensions - not just 2D matrices!
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
                zeros_info_dict[name] = get_zero_count(param.data)/param.data.numel()

        return zeros_info_dict

    def get_zero_count_in_mask(self, mask_dict):
        zeros_info_dict = dict()

        for name, data in mask_dict.items():
            zeros_info_dict[name] = get_zero_count(data)/data.numel()

        return zeros_info_dict

    def plot(self):
        percent_weights = [stat[self.PERCENTAGE_WEIGHT_MASKED] for stat in self.stats]
        accuracies = [stat[self.BEST_VALIDATION_ACCURACY] for stat in self.stats]
        plt.plot(percent_weights, accuracies)
        plt.xlabel('percentage of weights pruned')
        plt.ylabel('Early stopping val acc.')
        plt.savefig("graph.png")


class MNISTExperimentRunner(ExperimentRunner):
    def __init__(self, *args, **kwargs):
        super(MNISTExperimentRunner, self).__init__(*args, **kwargs)

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
            print('Best validation accuracy is: {} %'.format(self.get_stat(self.BEST_VALIDATION_ACCURACY)))
        self.update_stat(self.TEST_ACCURACY, test_accuracy)
        return test_accuracy

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


class ShuffleNetExperimentRunner(ExperimentRunner):
    def __init__(self, *args, **kwargs):
        super(ShuffleNetExperimentRunner, self).__init__(*args, **kwargs)

    def get_initial_mask(self):
        mask_dict = dict()
        for name, parameter in self.model.named_parameters():
            if 'weight' in name and ('conv' in name or 'fc' in name):
                mask_dict[name] = torch.ones(parameter.data.shape)

        return mask_dict

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
                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

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

        return validation_accuracy

    def test(self, input_size, test_dataloader):
        best_model = ShuffleNet(self.model.input_size, self.model.num_classes)
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load('temp.ckpt'))

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

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
            print('Best validation accuracy is: {} %'.format(self.get_stat(self.BEST_VALIDATION_ACCURACY)))
        self.update_stat(self.TEST_ACCURACY, test_accuracy)
        return test_accuracy

    def prune(self, mask_dict, prune_percent=0.1):
        # Use the best model obtained through early stopping. Weights are in the file temp.ckpt
        # TODO: Make this more elegant - do not hardcode the file name
        best_model = ShuffleNet(self.model.input_size, self.model.num_classes)
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load('temp.ckpt'))

        # We assume that all layers are pruned by the same percentage
        # Yes, we prune per layer, not globally

        for name, parameter in best_model.named_parameters():
            # TODO: Check if we should indeed ignore the bias
            if 'weight' in name:
                current_mask = mask_dict.get(name, None)
                if name == 'fc.weight':
                    # Last layer always has a different prune rate
                    new_mask = self.get_new_mask(prune_percent, parameter.data, current_mask)
                elif 'conv' in name:
                    new_mask = self.get_new_mask(prune_percent/2, parameter.data, current_mask)
                mask_dict[name] = new_mask

        self.update_stat(self.ZERO_PERCENTAGE_IN_MASKS, self.get_zero_count_in_mask(mask_dict))
        return mask_dict


class Conv2NetExperimentRunner(ExperimentRunner):
    def __init__(self, *args, **kwargs):
        super(Conv2NetExperimentRunner, self).__init__(*args, **kwargs)

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
                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

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

        return validation_accuracy

    def test(self, input_size, test_dataloader):
        best_model = Conv2Net(self.model.input_size, self.model.num_classes)
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load('temp.ckpt'))

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

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
            print('Best validation accuracy is: {} %'.format(self.get_stat(self.BEST_VALIDATION_ACCURACY)))
        self.update_stat(self.TEST_ACCURACY, test_accuracy)

        return test_accuracy

    def prune(self, mask_dict, prune_percent=0.1):
        # Use the best model obtained through early stopping. Weights are in the file temp.ckpt
        # TODO: Make this more elegant - do not hardcode the file name
        best_model = Conv2Net(self.model.input_size, self.model.num_classes)
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load('temp.ckpt'))

        # We assume that all layers are pruned by the same percentage
        # Yes, we prune per layer, not globally

        for name, parameter in best_model.named_parameters():
            # Prune the linear layers at prune_percent
            # Prune the output and convolutional layers at half that rate
            if 'weight' in name:
                current_mask = mask_dict.get(name, None)
                if 'conv' in name or 'output' in name:
                    new_mask = self.get_new_mask(prune_percent/2, parameter.data, current_mask)
                elif 'linear' in name:
                    new_mask = self.get_new_mask(prune_percent, parameter.data, current_mask)
                mask_dict[name] = new_mask

        self.update_stat(self.ZERO_PERCENTAGE_IN_MASKS, self.get_zero_count_in_mask(mask_dict))
        return mask_dict
