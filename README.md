# lottery-ticket-experiments
Experiments on the [lottery ticket hypothesis](https://arxiv.org/abs/1803.03635) for finding sparse trainable neural networks

## Objectives:

1. Reproduce winning lottery tickets on MNIST dataset (Lecun FCCN)
2. See if we can find winning lotter tickets for the ShuffleNet on CIFAR-10 (kinda)
3. Can we discover winning tickets faster by simply using less data? Will the tickets thus discovered continue to exhibit the lottery ticket pattern when retrained on the whole dataset? (yes!)

Please see our [report](https://lonesword.github.io/assets/lottery_ticket_team15_report.pdf) for more details on our experiment and the results we obtained


## How-To:

1. Install pytorch and torchvision. TODO: create a requirements.txt
2. `python experiment.py --experiment=mnist` to reproduce the MNIST baseline
3. `python experiment.py --experiment=shufflenet` to iteratively prune shufflenet using the CIFAR-10 dataset
