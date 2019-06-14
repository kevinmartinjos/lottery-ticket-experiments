def get_zero_count(matrix):
    # A utility function to count the number of zeroes in a 2-D matrix
    rows = len(matrix)
    cols = len(matrix[0])
    count = 0

    for i in range(0, rows):
        for j in range(0, cols):
            if not matrix[i][j]:
                count += 1

    return count


def apply_mask_dict_to_weight_dict(mask_dict, weight_dict):
    # mask_dict - a dictionary where keys are layer names (string) and values are masks (bytetensor) for that layer
    # weight_dict - a dictionary where keys are layer names and values are weights (tensor) for that layer
    # Applies the mask to the weight for each layer. This is done by simple multiplying the weight by the mask
    # (Hadamard product)
    # Since every value in the mask is either 0 or 1, this is equivalent to either letting the weight go unchanged or
    # setting it as 0
    weights_after_masking = dict()
    for layer_name, weight in weight_dict.items():
        mask = mask_dict[layer_name]
        weights_after_masking[layer_name] = weight * mask.float()

    return weights_after_masking

