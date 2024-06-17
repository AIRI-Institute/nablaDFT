# coding: utf-8

import random
import string

import numpy as np
import torch

_sqrt2 = np.sqrt(2)


def generate_id(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    """Used for creating a "unique" id for a run (almost impossible to generate the same twice)"""
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))


def compute_error_dict(predictions, data, loss_weights, max_errors, batch_size=1):
    error_dict = {"loss": 0.0}
    mask = data.get("mask")
    # print(data["energy"])

    for key in loss_weights.keys():
        if loss_weights[key] > 0:
            diff = predictions[key] - data[key][None, :, :]
            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))

            if mask is not None and key in [
                "full_hamiltonian",
                "core_hamiltonian",
                "overlap_matrix",
            ]:
                mse *= predictions[key].numel() / mask.sum()  # total number / nonzero elements
                mae *= predictions[key].numel() / mask.sum()

            if mae > max_errors[key]:
                error_dict[key + "_mae"] = torch.tensor(max_errors[key])
                error_dict[key + "_rmse"] = torch.tensor(_sqrt2 * max_errors[key])
            else:
                error_dict[key + "_mae"] = mae
                error_dict[key + "_rmse"] = torch.sqrt(mse)
                loss = mse + mae

            error_dict["loss"] = error_dict["loss"] + loss_weights[key] * loss
    return error_dict


def empty_error_dict(loss_weights, fill_value=0.0):
    """Returns an error dictionary filled with zeros"""
    error_dict = {"loss": fill_value}

    for key in loss_weights.keys():
        if loss_weights[key] > 0:
            error_dict[key + "_mae"] = fill_value
            error_dict[key + "_rmse"] = fill_value

    return error_dict
