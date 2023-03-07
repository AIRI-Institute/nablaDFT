# coding: utf-8

import sys
import argparse
import torch


def str2bool(s):
    """helper function used in order to support boolean command line arguments"""
    if s.lower() in ("true", "t", "1"):
        return True
    elif s.lower() in ("false", "f", "0"):
        return False
    else:
        return s


def parse_command_line_arguments():
    # declare parser
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._action_groups.pop()

    # argument for restarting runs
    args_restart = parser.add_argument_group("specification of a restart folder")
    args_restart.add_argument(
        "--restart",
        metavar="FOLDER",
        type=str,
        default=None,
        help="restart training from the given folder (all other arguments are ignored)",
    )

    # arguments for neural network architecture hyperparameters
    args_hyperparams = parser.add_argument_group(
        "neural network architecture hyperparameters"
    )
    args_hyperparams.add_argument(
        "--load_from",
        metavar="STR",
        type=str,
        default=None,
        help="initialize model from given pth file (other architecture hyperparameters are ignored)",
    )
    args_hyperparams.add_argument(
        "--activation",
        metavar="STR",
        type=str,
        default="swish",
        choices=["ssp", "swish"],
        help="which activation function to use (shifted softplus (ssp) or swish))",
    )
    args_hyperparams.add_argument(
        "--order",
        metavar="INT",
        type=int,
        default=2,
        help="angular order of the feature vectors",
    )
    args_hyperparams.add_argument(
        "--num_features",
        metavar="INT",
        type=int,
        default=32,
        help="dimensionality of feature vectors",
    )
    args_hyperparams.add_argument(
        "--num_basis_functions",
        metavar="INT",
        type=int,
        default=32,
        help="number of radial basis functions",
    )
    args_hyperparams.add_argument(
        "--num_modules",
        metavar="INT",
        type=int,
        default=3,
        help="number of modules used in the neural network (interaction iterations)",
    )
    args_hyperparams.add_argument(
        "--num_residual_pre_x",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for refining atomic feature vectors pre interaction",
    )
    args_hyperparams.add_argument(
        "--num_residual_post_x",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for refining atomic feature vectors post interaction",
    )
    args_hyperparams.add_argument(
        "--num_residual_pre_vi",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for refining interaction feature vectors pre interaction (channel i)",
    )
    args_hyperparams.add_argument(
        "--num_residual_pre_vj",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for refining interaction feature vectors pre interaction (channel j)",
    )
    args_hyperparams.add_argument(
        "--num_residual_post_v",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for refining interaction feature vectors post interaction",
    )
    args_hyperparams.add_argument(
        "--num_residual_output",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for refining output feature vectors",
    )
    args_hyperparams.add_argument(
        "--num_residual_pc",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for refining atomic feature vectors before constructing pair features (central atoms)",
    )
    args_hyperparams.add_argument(
        "--num_residual_pn",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for refining atomic feature vectors before constructing pair features (neighboring atoms)",
    )
    args_hyperparams.add_argument(
        "--num_residual_ii",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for predicting diagonal blocks (shared)",
    )
    args_hyperparams.add_argument(
        "--num_residual_ij",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for predicting off-diagonal blocks (shared)",
    )
    args_hyperparams.add_argument(
        "--num_residual_full_ii",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for predicting diagonal blocks (full hamiltonian)",
    )
    args_hyperparams.add_argument(
        "--num_residual_full_ij",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for predicting off-diagonal blocks (full hamiltonian)",
    )
    args_hyperparams.add_argument(
        "--num_residual_core_ii",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for predicting diagonal blocks (core hamiltonian)",
    )
    args_hyperparams.add_argument(
        "--num_residual_core_ij",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for predicting off-diagonal blocks (core hamiltonian)",
    )
    args_hyperparams.add_argument(
        "--num_residual_over_ij",
        metavar="INT",
        type=int,
        default=1,
        help="number of residual blocks for predicting off-diagonal blocks (overlap matrix)",
    )
    args_hyperparams.add_argument(
        "--basis_functions",
        metavar="STR",
        type=str,
        default="exp-bernstein",
        choices=["exp-bernstein", "exp-gaussian", "bernstein", "gaussian"],
        help="which type of basis functions to use",
    )
    args_hyperparams.add_argument(
        "--cutoff",
        metavar="FLOAT",
        type=float,
        default=7.937658158457616,
        help="cutoff radius for interactions (default corresponds to 15 Bohr)",
    )
    args_hyperparams.add_argument(
        "--orthonormal_basis",
        metavar="True|False",
        type=str2bool,
        default=False,
        choices=[True, False],
        help="use orthonormal basis (overlap matrix is identity) (will only work with appropriate reference data)",
    )

    # arguments for training
    args_training = parser.add_argument_group("training hyperparameters")
    args_training.add_argument(
        "--max_steps", metavar="INT", type=int, help="maximum number of training steps"
    )
    args_training.add_argument(
        "--datapath", metavar="STR", type=str, help="filepath to dataset"
    )
    args_training.add_argument(
        "--dataset_name", metavar="STR", type=str, help="name of the dataset"
    )
    args_training.add_argument(
        "--splits_file", metavar="STR", type=str, help="filepath to splitfile"
    )
    args_training.add_argument(
        "--subset", metavar="STR", type=str, default="", help="filepath to subset file"
    )
    args_training.add_argument(
        "--num_train", metavar="INT", type=int, help="size of training set"
    )
    args_training.add_argument(
        "--num_valid", metavar="INT", type=int, help="size of validation set"
    )
    args_training.add_argument(
        "--train_batch_size",
        metavar="INT",
        type=int,
        default=1,
        help="batch size for training",
    )
    args_training.add_argument(
        "--valid_batch_size",
        metavar="INT",
        type=int,
        default=1,
        help="batch size for validation",
    )
    args_training.add_argument(
        "--num_workers",
        metavar="INT",
        type=int,
        default=0,
        help="number of worker threads for preparing batches",
    )
    args_training.add_argument(
        "--split_seed",
        metavar="INT",
        type=int,
        default=42,
        help="seed for splitting the dataset in training, validation and test sets",
    )
    args_training.add_argument(
        "--optimizer",
        metavar="adam|amsgrad|sgd",
        type=str,
        default="sgd",
        choices=["adam", "amsgrad", "sgd"],
        help="optimizer used for training",
    )
    args_training.add_argument(
        "--lookahead_k",
        metavar="INT",
        type=int,
        default=5,
        help="Lookahead uses k steps (-1 -> no Lookahead is used)",
    )
    args_training.add_argument(
        "--learning_rate",
        metavar="FLOAT",
        type=float,
        default=1e-3,
        help="learning rate for the optimizer",
    )
    args_training.add_argument(
        "--decay_factor",
        metavar="FLOAT",
        type=float,
        default=0.5,
        help="learning rate is decayed by this factor whenever the validation loss does not improve after decay_patience intervals",
    )
    args_training.add_argument(
        "--decay_patience",
        metavar="INT",
        type=int,
        default=10,
        help="how many validation intervals have to be seen without improvement before the learning rate is decayed",
    )
    args_training.add_argument(
        "--stop_at_learning_rate",
        metavar="FLOAT",
        type=float,
        default=1e-1,
        help="when the learning rate gets lower than this value, training is stopped",
    )
    args_training.add_argument(
        "--epsilon",
        metavar="FLOAT",
        type=float,
        default=1e-8,
        help="epsilon for the optimizer (only relevant for Adam/AMSGrad)",
    )
    args_training.add_argument(
        "--beta1",
        metavar="FLOAT",
        type=float,
        default=0.9,
        help="beta1 for the optimizer (only relevant for Adam/AMSGrad)",
    )
    args_training.add_argument(
        "--beta2",
        metavar="FLOAT",
        type=float,
        default=0.999,
        help="beta2 for the optimizer (only relevant for Adam/AMSGrad)",
    )
    args_training.add_argument(
        "--momentum",
        metavar="FLOAT",
        type=float,
        default=0.0,
        help="momentum for the optimizer (only relevant for SGD)",
    )
    args_training.add_argument(
        "--full_hamiltonian_weight",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="weight of the full hamiltonian in the loss function",
    )
    args_training.add_argument(
        "--core_hamiltonian_weight",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="weight of the core hamiltonian in the loss function",
    )
    args_training.add_argument(
        "--overlap_matrix_weight",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="weight of the overlap matrix in the loss function",
    )
    args_training.add_argument(
        "--energy_weight",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="weight of the energy in the loss function",
    )
    args_training.add_argument(
        "--forces_weight",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="weight of the forces in the loss function",
    )
    args_training.add_argument(
        "--predict_energy",
        metavar="True|False",
        type=bool,
        default=False,
        help="calculate full energy",
    )
    args_training.add_argument(
        "--max_energy_error",
        metavar="FLOAT",
        type=float,
        default=0.1,
        help="for better stability at beginning of training: maximum allowed MAE in energy (higher errors are clamped)",
    )
    args_training.add_argument(
        "--max_forces_error",
        metavar="FLOAT",
        type=float,
        default=0.1,
        help="for better stability at beginning of training: maximum allowed MAE in forces (higher errors are clamped)",
    )
    args_training.add_argument(
        "--use_gradient_clipping",
        metavar="True|False",
        type=str2bool,
        default=False,
        choices=[True, False],
        help="use gradient clipping during training",
    )
    args_training.add_argument(
        "--clip_norm",
        metavar="FLOAT",
        type=float,
        default=1000.0,
        help="gradient clip norm (only when --use_gradient_clipping is active)",
    )
    args_training.add_argument(
        "--use_parameter_averaging",
        metavar="True|False",
        type=str2bool,
        default=True,
        choices=[True, False],
        help="keep exponential moving average of model parameters (might boost convergence speed)",
    )
    args_training.add_argument(
        "--ema_decay",
        metavar="FLOAT",
        type=float,
        default=0.999,
        help="decay rate used for exponential moving average of parameters",
    )
    args_training.add_argument(
        "--ema_start_epoch",
        metavar="INT",
        type=int,
        default=0,
        help="starts exponential moving average of parameters only after the specified epoch is reached",
    )
    args_training.add_argument(
        "--weight_decay",
        metavar="FLOAT",
        type=float,
        default=0.0,
        help="regularization term for weights",
    )
    args_training.add_argument(
        "--use_gpu",
        metavar="True|False",
        type=str2bool,
        default=True,
        choices=[True, False],
        help="use GPU(s) for training (if available)",
    )
    args_training.add_argument(
        "--nproc_per_node",
        metavar="INT",
        type=int,
        default=1,
        help="use GPU(s) for training (if available)",
    )
    args_training.add_argument(
        "--nnodes",
        metavar="INT",
        type=int,
        default=1,
        help="use GPU(s) for training (if available)",
    )
    args_training.add_argument(
        "--local_rank",
        metavar="INT",
        type=int,
        default=0,
        help="use GPU(s) for training (if available)",
    )
    args_training.add_argument(
        "--node_rank",
        metavar="INT",
        type=int,
        default=0,
        help="use GPU(s) for training (if available)",
    )

    # arguments for logging and checkpoints
    args_logging = parser.add_argument_group("logging and checkpoints")
    args_logging.add_argument(
        "--write_parameter_summaries",
        metavar="True|False",
        type=str2bool,
        default=False,
        choices=[True, False],
        help="write summaries for parameters",
    )
    args_logging.add_argument(
        "--validation_interval",
        metavar="INT",
        type=int,
        default=1,
        help="perform model validation after every INT steps",
    )
    args_logging.add_argument(
        "--summary_interval",
        metavar="INT",
        type=int,
        default=1,
        help="log summaries after every INT steps",
    )
    args_logging.add_argument(
        "--checkpoint_interval",
        metavar="INT",
        type=int,
        default=1,
        help="write checkpoints after every INT steps",
    )
    args_logging.add_argument(
        "--keep_checkpoints",
        metavar="INT",
        type=int,
        default=0,
        help="keep X checkpoints older than the latest checkpoint (-1 keeps all checkpoints)",
    )

    # misc arguments
    args_misc = parser.add_argument_group("miscelleaneous")
    args_misc.add_argument(
        "--dtype",
        metavar="torch.float32|torch.float64",
        type=str,
        default="torch.float32",
        choices=["torch.float32", "torch.float64"],
        help="floating point type used during training",
    )

    # actually parse command line arguments
    if len(sys.argv) == 1:  # no arguments were specified, print help message
        args = parser.parse_args(["--help"])
    else:
        args = parser.parse_args()
        # convert dtype argument to the proper torch type
        if args.dtype == "torch.float32":
            args.dtype = torch.float32
        elif args.dtype == "torch.float64":
            args.dtype = torch.float64

        # necessary because None is not properly by argparse (special case)
        if args.restart == "None":
            args.restart = None
        if args.load_from == "None":
            args.load_from = None

    return args
