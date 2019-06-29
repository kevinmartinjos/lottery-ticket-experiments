import torch


def get_zero_count(matrix):
    # A utility function to count the number of zeroes in a 2-D matrix
    return torch.sum(matrix == 0)


def apply_mask_dict_to_weight_dict(mask_dict, weight_dict):
    # mask_dict - a dictionary where keys are layer names (string) and values are masks (bytetensor) for that layer
    # weight_dict - a dictionary where keys are layer names and values are weights (tensor) for that layer
    # Applies the mask to the weight for each layer. This is done by simple multiplying the weight by the mask
    # (Hadamard product)
    # Since every value in the mask is either 0 or 1, this is equivalent to either letting the weight go unchanged or
    # setting it as 0
    weights_after_masking = dict()
    for layer_name, mask in mask_dict.items():
        weight = weight_dict[layer_name]
        # The mask should be copied to the cpu since `weights_after_masking` dict is always stored in memory, and not the GPU
        weights_after_masking[layer_name] = weight * mask.cpu().float()

    return weights_after_masking

