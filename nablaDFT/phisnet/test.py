# coding: utf-8
import logging

import numpy as np
import torch
from nn import NeuralNetwork
from tqdm import tqdm
from train import compute_error_dict, empty_error_dict
from training import parse_command_line_arguments

from nablaDFT.dataset import NablaDFT

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    # read arguments and initialize results collectors
    args = parse_command_line_arguments()

    if args.dtype == "torch.float32":
        args.dtype = torch.float32
    elif args.dtype == "torch.float64":
        args.dtype = torch.float64

    # determine weights of different quantities for scaling loss
    loss_weights = dict()
    loss_weights["full_hamiltonian"] = args.full_hamiltonian_weight
    loss_weights["core_hamiltonian"] = args.core_hamiltonian_weight
    loss_weights["overlap_matrix"] = args.overlap_matrix_weight
    loss_weights["energy"] = args.energy_weight
    loss_weights["forces"] = args.forces_weight

    # if energies/forces are used for training, the extreme errors
    # at the beginning of training usually lead to NaNs. For this
    # reason gradients are only allowed to flow through loss terms
    # if the MAE is smaller than a certain threshold.
    max_errors = {
        "full_hamiltonian": np.inf,
        "core_hamiltonian": np.inf,
        "overlap_matrix": np.inf,
        "energy": args.max_energy_error,
        "forces": args.max_forces_error,
    }

    # # determine whether GPU is used for training
    use_gpu = args.use_gpu and torch.cuda.is_available()

    # ---------------- DATA ----------------
    # loaded = np.load(os.path.join(os.path.dirname(args.load_from), 'datasplits.npz'))

    logging.info("Loading " + str(args.dataset_name) + " from " + str(args.datapath) + "...")
    dataset = NablaDFT("Hamiltonian", args.datapath, args.dataset_name)
    batch_size = args.valid_batch_size

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        collate_fn=lambda batch: test_dataset.collate_fn(batch, return_filtered=True),
    )

    # model
    logging.info("Loading model... %s" % args.load_from)
    model = NeuralNetwork(load_from=args.load_from)

    # determine what should be calculated based on loss weights
    tmp = (loss_weights["energy"] > 0) or (loss_weights["forces"] > 0)
    model.calculate_full_hamiltonian = (loss_weights["full_hamiltonian"] > 0) or tmp
    model.calculate_core_hamiltonian = (loss_weights["core_hamiltonian"] > 0) or tmp
    model.calculate_overlap_matrix = ((loss_weights["overlap_matrix"] > 0) or tmp) and not args.orthonormal_basis

    model.calculate_energy = loss_weights["energy"] > 0
    model.calculate_energy = False

    model.calculate_forces = loss_weights["forces"] > 0

    model.to(args.dtype)

    # send model to GPU (if use_gpu is True)
    logging.info("Should use GPU? " + str(use_gpu))

    if use_gpu:
        model.cuda()

    logging.info("Eval mode...")

    test_errors = empty_error_dict(loss_weights)  # reset valid error metrics
    model.eval()  # sets model to evaluation mode

    """
        python test.py  @2022-03-17_n1CF6X5M/args.txt --load_from 2022-03-17_n1CF6X5M/best_n1CF6X5M.pth
    """

    with torch.no_grad():
        for test_batch_num, data in tqdm(
            enumerate(test_data_loader),
            "batches, test set",
            total=len(test_dataset) // batch_size + 1,
        ):
            # send data to GPU
            if use_gpu:
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda()

            # forward step
            predictions = model(data)

            # compute error metrics
            errors = compute_error_dict(predictions, data, loss_weights, max_errors, batch_size=batch_size)

            # update valid_errors (running average)
            for key in test_errors.keys():
                test_errors[key] += (errors[key].item() - test_errors[key]) / (test_batch_num + 1)

        # construct message for logging
        message = ""
        for key in test_errors.keys():
            message += key + ": %.9f" % test_errors[key] + "\n"

        print(message)

        progress_string = ""

        for key in loss_weights.keys():
            if loss_weights[key] > 0:
                progress_string += "\n  " + key + ":\n"
                progress_string += "    test: %10.8f" % test_errors[key + "_mae"]

        print(progress_string)
