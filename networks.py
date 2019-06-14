# The plan:
# Write a base class that can take in a mask and a pre-init
from torch import nn
import hyperparameter_presets


class LotteryExperimentNetwork(nn.Module):
    def __init__(self, pre_init=None, mask=None):
        # pre_init and mask should both be dicts.
        # The keys should be the name of the layers
        super(LotteryExperimentNetwork, self).__init__()
        self.mask_dict = mask
        self.pre_init = pre_init
        self.apply_pre_init(pre_init)

        # Just making sure that the masks are applied before each forward pass
        # Perhaps we just have to do it once in the beginning and that's all
        # TODO: Check if it's necessary to apply the mask before every forward call
        # Intuition:
        # 1. The mask sets some weights as zero
        # 2. If those weights are zero, their gradient would also be zero since they did not contribute to the loss at
        # all
        self.register_forward_pre_hook(self.apply_mask)

    def apply_pre_init(self, pre_init):
        # pre_init is a dict. Keys are strings that represent layer names. Values are weights
        raise NotImplementedError

    def apply_mask(self, *args, **kwargs):
        mask_dict = self.mask_dict
        if mask_dict is None:
            return
        else:
            for name, param in self.named_parameters():
                mask = mask_dict.get(name)
                self.apply_mask_to_layer(name, mask)

    def apply_mask_to_layer(self, layer_name, mask):
        if mask is None:
            return
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError


class FullyConnectedMNIST(LotteryExperimentNetwork):
    def __init__(self, input_size, hidden_sizes, num_classes, pre_init=None, mask=None):
        super(FullyConnectedMNIST, self).__init__()
        layers = []

        # So that we can iterate through the layer_sizes
        layer_sizes = [input_size] + hidden_sizes + [num_classes]

        for i in range(0, len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def weights_init(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0.0, 1e-3)
            m.bias.data.fill_(0.)

    def apply_pre_init(self, pre_init):
        # TODO: This should be implemented in the super class

        # No pre_init. Hence randomly initialize the weights for the Linear layers
        if pre_init is None:
            self.apply(self.weights_init)

    def forward(self, x):
        return self.layers(x)
