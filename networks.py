# The plan:
# Write a base class that can take in a mask and a pre-init
from torch import nn

from utils import get_zero_count


class NeuralNetUtilsMixin:
    def retrieve_initial_weights(self):
        initial_weights = dict()

        for name, parameter in self.named_parameters():
            if name.endswith('weight'):
                initial_weights[name] = parameter.data.clone()

        return initial_weights


class LotteryExperimentNetwork(nn.Module, NeuralNetUtilsMixin):
    def __init__(self, pre_init=None, mask_dict=None):
        # pre_init and mask should both be dicts.
        # The keys should be the name of the layers
        super(LotteryExperimentNetwork, self).__init__()
        self.mask_dict = mask_dict
        self.pre_init = pre_init
        self.apply_pre_init(pre_init)

        # Just making sure that the masks are applied before each forward pass
        # Perhaps we just have to do it once in the beginning and that's all
        # TODO: Check if it's necessary to apply the mask before every forward call
        # Intuition:
        # 1. The mask sets some weights as zero
        # 2. If those weights are zero, their gradient would also be zero since they did not contribute to the loss at
        # all
        self.register_forward_pre_hook(self.apply_mask_to_model)

    def apply_pre_init(self, pre_init):
        # pre_init is a dict. Keys are strings that represent layer names. Values are weights
        for name, param in self.named_parameters():
            if pre_init.get(name) is not None:
                param.data = pre_init[name]

    def apply_mask_to_model(self, *args, **kwargs):
        mask_dict = self.mask_dict
        if mask_dict is None:
            return
        else:
            for name, param in self.named_parameters():
                mask = mask_dict.get(name)
                self.apply_mask_to_layer(param, mask)

    def apply_mask_to_layer(self, param, mask):
        # mask - tensor byte
        if mask is None:
            return
        else:
            param.data = param.data * mask.float()

    def get_percent_weights_masked(self):
        if not self.mask_dict:
            return 0

        total_weights = 0
        total_weights_masked = 0

        for layer_name, mask in self.mask_dict.items():
            total_weights += mask.shape[0] * mask.shape[1]
            total_weights_masked += get_zero_count(mask)

        return total_weights_masked/total_weights

    def forward(self, *args):
        raise NotImplementedError


class FullyConnectedMNIST(LotteryExperimentNetwork):
    def __init__(self, input_size, hidden_sizes, num_classes, pre_init=None, mask_dict=None):
        super(FullyConnectedMNIST, self).__init__(pre_init=pre_init, mask_dict=mask_dict)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.pre_init = pre_init
        self.mask_dict = mask_dict
        self.layers = self.create_layers()
        self.initial_weights = self.retrieve_initial_weights()

    def create_layers(self):
        layers = []

        # So that we can iterate through the layer_sizes
        layer_sizes = [self.input_size] + self.hidden_sizes

        for i in range(0, len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())

        self.output_layer = nn.Linear(layer_sizes[i+1], self.num_classes)
        layers.append(self.output_layer)
        self.output_relu = nn.ReLU()
        layers.append(self.output_relu)

        return nn.Sequential(*layers)

    def weights_init(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0.0, 1e-3)
            m.bias.data.fill_(0.)

    def apply_pre_init(self, pre_init):
        # TODO: This should be implemented in the super class

        # No pre_init. Hence randomly initialize the weights for the Linear layers
        if pre_init is None:
            self.apply(self.weights_init)
        else:
            super(FullyConnectedMNIST, self).apply_pre_init(pre_init)

    def forward(self, x):
        return self.layers(x)
