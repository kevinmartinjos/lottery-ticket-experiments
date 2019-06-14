import time
import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms

import hyperparameter_presets

# A helper class that takes a model and dataset, and runs the experiment on it.
from networks import FullyConnectedMNIST


class ExperimentRunner:
    TRAINING_DURATION_SECONDS = "training_duration_seconds"
    FINAL_VALIDATION_ACCURACY = "final_validation_accuracy"
    DEVICE = "device"

    def __init__(self, model, num_epochs=10, batch_size=200, learning_rate=1e-3, learning_rate_decay=0.95):
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
        for epoch in range(self.num_epochs):
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

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))
            lr = self.learning_rate * self.learning_rate_decay
            self.update_lr(optimizer, lr)
            validation_accuracy = self.validate(input_size, validation_dataloader)

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
            print('Validation accuracy is: {} %'.format(validation_accuracy))

        return validation_accuracy

    def test(self):
        pass


def mnist_experiment():
    num_classes = hyperparameter_presets.FULLY_CONNECTED_MNIST['num_classes']
    input_size = hyperparameter_presets.FULLY_CONNECTED_MNIST['input_size']
    hidden_sizes = hyperparameter_presets.FULLY_CONNECTED_MNIST['hidden_sizes']

    # Temporary parameters. Should probably move this to the hyper parameters file as well
    num_training = 58000
    num_validation = 2000
    batch_size = 200

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
    model.cuda()
    experiment = ExperimentRunner(model)
    experiment.train(input_size, mnist_train_loader, mnist_val_loader)
    experiment.print_stats()

if __name__ == "__main__":
    mnist_experiment()

