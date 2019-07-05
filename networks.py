# The plan:
# Write a base class that can take in a mask and a pre-init
from torch import nn

import math
import torch
import torch.nn.functional as F

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

        # Just making sure that the masks are applied before each forward pass
        # Perhaps we just have to do it once in the beginning and that's all
        # TODO: Check if it's necessary to apply the mask before every forward call
        # Intuition:
        # 1. The mask sets some weights as zero
        # 2. If those weights are zero, their gradient would also be zero since they did not contribute to the loss at
        # all
        self.register_forward_pre_hook(self.apply_mask_to_model)

    def apply_pre_init(self, pre_init):
        raise NotImplementedError

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
            total_weights += mask.numel()
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
        self.apply_pre_init(pre_init)
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
            self.load_state_dict(pre_init, strict=False)

    def forward(self, x):
        return self.layers(x)


class ShuffleNetUnit(nn.Module):
    def __init__(self, input_size, output_size, stride=1, do_concat=False):
        super(ShuffleNetUnit, self).__init__()

        # TODO: Create a custom exception class for this and move it to another file
        if stride not in [1, 2]:
            raise Exception("ShuffleNetUnit stride should be either 1 or 2")

        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.g = 1

        # See the shuffleNetUnit figures (figure 2) in the paper. If stride = 2, we should do a avg pooling and concat
        self.do_concat = do_concat

        self.create_layers()

    def create_layers(self):
        """
        These are the layers (in order):
            - 1x1 group convolution
            - channel shuffle (not a layer, so would be omitted here)
            - Depthwise convolution
            - 1x1 group convolution
            - Batch normalization and ReLus in between the abotorch.Size([100, 24, 7, 7])ve layers as applicable
        """

        # bottle neck channel = input channel / 4, as the paper did
        # TODO: Figure out why we need this at all
        neck_channel_size = int(self.output_size/4)
        self.gconv1 = nn.Conv2d(self.input_size, neck_channel_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(neck_channel_size)

        self.dwconv = nn.Conv2d(
            neck_channel_size, neck_channel_size, groups=neck_channel_size, stride=self.stride, kernel_size=3,
            padding=1, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(neck_channel_size)

        if self.do_concat:
            self.gconv2 = nn.Conv2d(
                neck_channel_size, self.output_size - self.input_size, groups=1, kernel_size=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(self.output_size - self.input_size)
        else:
            # TODO: Figure out why we need self.output_size - self.input_size for the stride=2 unit
            self.gconv2 = nn.Conv2d(
                neck_channel_size, self.output_size, groups=1, kernel_size=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(self.output_size)

        # Not shuffling
        # for channel shuffle operation
        self.n = int(neck_channel_size / self.g)  # self.g is hard coded to be 1 for the time being
        # self.g, self.n = g, neck_channel / g
        # assert self.n == int(self.n), "error shape to shuffle"

    def forward(self, inputs):
        x = F.relu(self.bn1(self.gconv1(inputs)))

        # channel shuffle
        n, c, w, h = x.shape
        x = x.view(n, self.g, self.n, w, h)
        x = x.transpose_(1, 2).contiguous()
        x = x.view(n, c, w, h)

        x = self.bn_dw(self.dwconv(x))
        x = self.bn2(self.gconv2(x))

        if self.do_concat:
            # Shortcut refers to the shortcut path when stride = 2 (refer to figure 2 in the paper)
            shortcut = F.avg_pool2d(inputs, 2)
            return F.relu(torch.cat((x, shortcut), dim=1))
        else:
            return F.relu(x + inputs)


class ShuffleNet(LotteryExperimentNetwork):
    """
    Draws heavy inspiration from: https://github.com/tinyalpha/shuffleNet-cifar10
    """
    def __init__(self, input_size, num_classes, pre_init=None, mask_dict=None):
        super(ShuffleNet, self).__init__(pre_init=pre_init, mask_dict=mask_dict)

        # TODO: This part of the code is similar for MNIST as well. Abstract this away?
        self.input_size = input_size
        self.num_classes = num_classes
        self.pre_init = pre_init
        self.mask_dict = mask_dict
        self.create_layers()
        self.apply_pre_init(pre_init)
        self.initial_weights = self.retrieve_initial_weights()
        self.g = 1

    def create_layers(self):
        # Refer to table 1 in the shuffle net paper. The layers are created in the order mentioned in the table
        self.conv_1 = nn.Conv2d(3, 24, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(24)
        self.max_pool_1 = nn.MaxPool2d(3, stride=2)  # TODO: Fix stride and padding if necessary

        self.g = 1  # Redundant, I know
        c2_size = 144
        c3_size = 2 * c2_size
        c4_size = 4 * c2_size

        self.stage_2 = self.build_stage(24, c2_size, repeat=3)
        self.stage_3 = self.build_stage(c2_size, c3_size, repeat=7)
        self.stage_4 = self.build_stage(c3_size, c4_size, repeat=3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4_size, self.num_classes)

    def build_stage(self, input_size, output_size, repeat=1):
        # According to the paper, each stage consists of 2 parts:
        # 1. A shuffleNetUnit with stride=2 and repeat = x1, corresponding to (c) in figure 2
        # 1. A shuffleNetUnit with stride=1 and repeat = x, corresponding to (b) in figure 2

        stage = [ShuffleNetUnit(input_size, output_size, stride=2, do_concat=True)]
        for i in range(0, repeat):
            stage.append(ShuffleNetUnit(output_size, output_size, stride=1, do_concat=False))

        return nn.Sequential(*stage)

    def weights_init(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0.0, 1e-3)
            m.bias.data.fill_(0.)
        if type(m) == nn.Conv2d:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif type(m) == nn.BatchNorm2d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def apply_pre_init(self, pre_init):
        # TODO: This should be implemented in the super class

        # No pre_init. Hence randomly initialize the weights for the Linear layers
        if pre_init is None:
            self.apply(self.weights_init)
        else:
            self.load_state_dict(pre_init, strict=False)

    def forward(self, inputs):
        # first conv layer
        x = F.relu(self.bn_1(self.conv_1(inputs)))
        # x = self.max_pool_1(x) TODO: Uncomment this?

        # bottlenecks
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)

        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class Conv2Net(LotteryExperimentNetwork):
    def __init__(self, input_size, num_classes, pre_init=None, mask_dict=None):
        super(Conv2Net, self).__init__(pre_init=pre_init, mask_dict=mask_dict)
        self.input_size = input_size
        self.num_classes = num_classes
        self.pre_init = pre_init
        self.mask_dict = mask_dict
        self.layers = self.create_layers()
        self.apply_pre_init(pre_init)
        self.initial_weights = self.retrieve_initial_weights()

    def create_layers(self):
        layers = []

        # So that we can iterate through the layer_sizes
        self.conv_1 = nn.Conv2d(3, 64, 3)
        self.conv_2 = nn.Conv2d(64, 64, 3)
        self.max_pool = nn.MaxPool2d(3, stride=2)

        self.linear_1 = nn.Linear(64, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, self.num_classes)

    def weights_init(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0.0, 1e-3)
            m.bias.data.fill_(0.)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)

    def apply_pre_init(self, pre_init):
        # TODO: This should be implemented in the super class

        # No pre_init. Hence randomly initialize the weights for the Linear layers
        if pre_init is None:
            self.apply(self.weights_init)
        else:
            self.load_state_dict(pre_init, strict=False)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.max_pool(x)
        x = x.view(-1, 64)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.output(x)
        return x
