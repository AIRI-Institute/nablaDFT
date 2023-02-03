#!/usr/bin/env python3

import time
import math
import torch
import logging
import multiprocessing
import numpy as np
from datetime import datetime

from tensorboardX import SummaryWriter
from torch.nn.functional import softplus

from nn import NeuralNetwork
from training import ExponentialMovingAverage, Lookahead
from training import seeded_random_split, parse_command_line_arguments, generate_id, empty_error_dict, compute_error_dict

import sys
sys.path.append('../')
from dataset.nablaDFT import nablaDFT

# without this, some things from torch don't
# work correctly in newer versions of python
multiprocessing.set_start_method('fork')


if __name__ == "__main__":

    import os

    logging.basicConfig()
    logging.getLogger().setLevel(logging.NOTSET)

    """
    ################################################
    ################ INITIALIZATION ################
    ################################################
    """
    # read arguments
    args = parse_command_line_arguments()

    # no restart directory specified
    if args.restart is None:

        ID = generate_id()  # generate "unique" id for the run (very unlikely that two runs will have the same ID)
        directory = datetime.utcnow().strftime("%Y-%m-%d_") + ID  # generate directory name
        directory = os.path.join("khrabrov/logs", directory)
        checkpoint_dir = os.path.join(directory, 'checkpoints')  # checkpoint directory

        # create directories
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # write command line arguments to file (useful for reproducibility)
        with open(os.path.join(directory, 'args.txt'), 'w') as f:
            for key in args.__dict__.keys():
                if isinstance(args.__dict__[key], list):  # special case for list input
                    for entry in args.__dict__[key]:
                        f.write('--' + key + '=' + str(entry) + "\n")
                else:
                    f.write('--' + key + '=' + str(args.__dict__[key]) + "\n")

        checkpoint = None
        latest_checkpoint = 0

    # restarts run from latest checkpoint
    else:
        directory = args.restart  # load directory name
        checkpoint_dir = os.path.join(directory, 'checkpoints')  # checkpoint directory

        # load latest checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'), map_location='cpu')
        latest_checkpoint = checkpoint['epoch']
        ID = checkpoint['ID']  # load ID
        # args = checkpoint['args']  # overwrite args

    # determine whether GPU is used for training
    use_gpu = args.use_gpu and torch.cuda.is_available()

    # load dataset(s)
    logging.info("Loading " + args.dataset_name + "...")
    dataset = nablaDFT('Hamiltonian', args.datapath, args.dataset_name)
    # split into train/valid/test

    train_dataset, valid_dataset, test_dataset = seeded_random_split(dataset, [args.num_train, args.num_valid,
                                                                               len(dataset) - (
                                                                                       args.num_train + args.num_valid)],
                                                                     seed=args.split_seed)

    logging.info(str([args.num_train, args.num_valid, len(dataset) - (args.num_train + args.num_valid)]))

    # train_dataset, valid_dataset, test_dataset = file_split(dataset, args.splits_file)

    # save indices for splits
    np.savez(os.path.join(directory, 'datasplits.npz'), train=train_dataset.indices, valid=valid_dataset.indices,
             test=test_dataset.indices)

    # determine weights of different quantities for scaling loss
    loss_weights = dict()
    loss_weights['full_hamiltonian'] = args.full_hamiltonian_weight
    loss_weights['core_hamiltonian'] = args.core_hamiltonian_weight
    loss_weights['overlap_matrix'] = args.overlap_matrix_weight
    loss_weights['energy'] = args.energy_weight
    loss_weights['forces'] = args.forces_weight

    # if energies/forces are used for training, the extreme errors
    # at the beginning of training usually lead to NaNs. For this
    # reason gradients are only allowed to flow through loss terms
    # if the MAE is smaller than a certain threshold.
    max_errors = {'full_hamiltonian': np.inf,
                  'core_hamiltonian': np.inf,
                  'overlap_matrix': np.inf,
                  'energy': args.max_energy_error,
                  'forces': args.max_forces_error}

    # prepare data loaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=use_gpu,
                                                    collate_fn=lambda batch: dataset.collate_fn(batch))
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=use_gpu,
                                                    collate_fn=lambda batch: dataset.collate_fn(batch))

    # define model
    if args.load_from is None:
        model = NeuralNetwork(
            max_orbitals=dataset.max_orbitals,
            order=args.order,
            num_features=args.num_features,
            num_basis_functions=args.num_basis_functions,
            num_modules=args.num_modules,
            num_residual_pre_x=args.num_residual_pre_x,
            num_residual_post_x=args.num_residual_post_x,
            num_residual_pre_vi=args.num_residual_pre_vi,
            num_residual_pre_vj=args.num_residual_pre_vj,
            num_residual_post_v=args.num_residual_post_v,
            num_residual_output=args.num_residual_output,
            num_residual_pc=args.num_residual_pc,
            num_residual_pn=args.num_residual_pn,
            num_residual_ii=args.num_residual_ii,
            num_residual_ij=args.num_residual_ij,
            num_residual_full_ii=args.num_residual_full_ii,
            num_residual_full_ij=args.num_residual_full_ij,
            num_residual_core_ii=args.num_residual_core_ii,
            num_residual_core_ij=args.num_residual_core_ij,
            num_residual_over_ij=args.num_residual_over_ij,
            basis_functions=args.basis_functions,
            cutoff=args.cutoff,
            activation=args.activation)
    else:
        model = NeuralNetwork(load_from=args.load_from)

    # determine what should be calculated based on loss weights
    tmp = (loss_weights['energy'] > 0) or (loss_weights['forces'] > 0)
    model.calculate_full_hamiltonian = (loss_weights['full_hamiltonian'] > 0) or tmp
    model.calculate_core_hamiltonian = (loss_weights['core_hamiltonian'] > 0) or tmp
    model.calculate_overlap_matrix = ((loss_weights['overlap_matrix'] > 0) or tmp) and not args.orthonormal_basis
    if args.predict_energy:
        model.predict_energy = 1
        model.calculate_energy = 0
    else:
        model.predict_energy = 0
        model.calculate_energy = loss_weights['energy'] > 0
    model.calculate_forces = loss_weights['forces'] > 0
    # convert the model to the correct dtype
    model.to(args.dtype)

    # send model to GPU (if use_gpu is True)
    if use_gpu:
        model.cuda()

    module = model
    # for keeping an exponential moving average of the model parameters (usually leads to better models)
    if args.use_parameter_averaging:
        exponential_moving_average = ExponentialMovingAverage(module, decay=args.ema_decay,
                                                              start_epoch=args.ema_start_epoch)
    else:
        exponential_moving_average = None

    # build list of parameters to optimize (with or without weight decay)
    parameters = []
    weight_decay_parameters = []
    for name, param in module.named_parameters():
        if 'weight' in name and not 'radial_fn' in name and not 'embedding' in name:
            weight_decay_parameters.append(param)
        else:
            parameters.append(param)

    parameter_list = [
        {'params': parameters},
        {'params': weight_decay_parameters, 'weight_decay': float(args.weight_decay)}]

    # choose optimizer
    if args.optimizer == 'adam':  # Adam
        logging.info("using Adam optimizer")
        optimizer = torch.optim.AdamW(parameter_list, lr=args.learning_rate, eps=args.epsilon,
                                      betas=(args.beta1, args.beta2), weight_decay=0.0)
    elif args.optimizer == 'amsgrad':  # AMSGrad
        logging.info("using AMSGrad optimizer")
        optimizer = torch.optim.AdamW(parameter_list, lr=args.learning_rate, eps=args.epsilon,
                                      betas=(args.beta1, args.beta2), weight_decay=0.0, amsgrad=True)
    elif args.optimizer == 'sgd':  # Stochastic Gradient Descent
        logging.info("using Stochastic Gradient Descent optimizer")
        optimizer = torch.optim.SGD(parameter_list, lr=args.learning_rate, momentum=args.momentum, weight_decay=0.0)

    # initialize Lookahead
    if args.lookahead_k > 0:
        optimizer = Lookahead(optimizer, k=args.lookahead_k)

    # learning rate scheduler (decays learning rate if validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.decay_factor,
                                                           patience=args.decay_patience)

    # restore state from checkpoint
    if checkpoint is not None:  # no checkpoint is specified

        step = checkpoint['step']
        epoch = checkpoint['epoch']
        best_errors = checkpoint['best_errors']
        valid_errors = checkpoint['valid_errors']
        module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if exponential_moving_average is not None:
            checkpoint_ema = checkpoint['exponential_moving_average']
            for key in exponential_moving_average.ema.keys():
                with torch.no_grad():
                    exponential_moving_average.ema[key].data.copy_(checkpoint_ema[key].data)

    # or initialize step/epoch to 0 and errors to infinity
    else:
        step = 0
        epoch = 0
        best_errors = empty_error_dict(loss_weights, fill_value=math.inf)
        valid_errors = empty_error_dict(loss_weights, fill_value=math.inf)

    # create summary writer for tensorboard
    summary = SummaryWriter(logdir=os.path.join(directory, './logs/'), purge_step=step)

    """
    ###############################################
    ################ TRAINING LOOP ################
    ###############################################
    """
    if use_gpu:
        logging.info("Training on " + str(torch.cuda.device_count()) + " GPUs:")
    else:
        logging.info("Training on the CPU:")

    # initialize train metrics
    if args.use_gradient_clipping:
        gradient_norm = 0

    train_errors = empty_error_dict(loss_weights)  # reset train error metrics
    train_batch_num = -1

    # initialize state
    model.train()
    train_iterator = iter(train_data_loader)
    new_valid = False
    new_best = False

    start_time = time.time()
    oom = 0
    while step < args.max_steps + 1:

        # get the next batch
        try:
            data = next(train_iterator)
        except StopIteration:
            epoch += 1
            train_iterator = iter(train_data_loader)
            continue
        train_batch_num += 1

        # send data to GPU
        if use_gpu:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # with torch.autograd.set_detect_anomaly(True): #TODO!!! TURN THIS OFF AGAIN
        try:
            # forward step
            predictions = model(data)

            # compute error metrics
            errors = compute_error_dict(predictions, data, loss_weights, max_errors, logging)
            # backward step
            errors['loss'].backward()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom += 1
                print('| WARNING: ran out of memory, skipping batch')
                logging.debug(data['molecule_size'])
                if oom >= 10:
                    raise e
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # apply gradient clipping
        if args.use_gradient_clipping:
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), args.clip_norm)
            gradient_norm += (norm - gradient_norm) / (train_batch_num + 1)

        # optimization step
        optimizer.step()

        # update parameter averages
        if args.use_parameter_averaging:
            exponential_moving_average(epoch)

        # update train_errors (running average)
        for key in train_errors.keys():
            train_errors[key] += (errors[key].item() - train_errors[key]) / (train_batch_num + 1)

        # run validation each validation_interval
        if step % args.validation_interval == 0:
            # this is a signal to the summary writer
            new_valid = True

            # swap to exponentially averaged parameters for validation
            if args.use_parameter_averaging:
                exponential_moving_average.swap()

            # run once over the validation set
            valid_errors = empty_error_dict(loss_weights)  # reset valid error metrics
            model.eval()  # sets model to evaluation mode
            with torch.no_grad():
                for valid_batch_num, data in enumerate(valid_data_loader):
                    # send data to GPU
                    if use_gpu:
                        for key in data.keys():
                            if isinstance(data[key], torch.Tensor):
                                data[key] = data[key].cuda()

                    # forward step
                    predictions = model(data)

                    # compute error metrics
                    errors = compute_error_dict(predictions, data, loss_weights, max_errors, logging)

                    # update valid_errors (running average)
                    for key in valid_errors.keys():
                        valid_errors[key] += (errors[key].item() - valid_errors[key]) / (valid_batch_num + 1)

            # pass validation loss to learning rate scheduler
            scheduler.step(metrics=valid_errors['loss'])

            # save if it outperforms previous best
            if valid_errors['loss'] < best_errors['loss']:
                new_best = True
                best_errors = valid_errors
                module.save(os.path.join(directory, 'best_' + str(ID) + '.pth'))
                # construct message for logging
                message = ''
                for key in best_errors.keys():
                    message += key + ': %.9f' % best_errors[key] + '\n'
                summary.add_text('best models', message, step)

            # swap back to original parameters for training
            if args.use_parameter_averaging:
                exponential_moving_average.swap()

            # set model back to training mode
            model.train()

        # write summary to console
        if step % args.summary_interval == 0:
            # write error summaries
            for key in train_errors.keys():
                summary.add_scalar(key + '/train', train_errors[key], step)

            if new_valid:
                for key in valid_errors.keys():
                    summary.add_scalar(key + '/valid', valid_errors[key], step)
                new_valid = False

            if new_best:
                for key in best_errors.keys():
                    summary.add_scalar(key + '/best', best_errors[key], step)
                new_best = False

            if args.use_gradient_clipping:
                summary.add_scalar('gradient/norm', gradient_norm, step)

            # write summaries for scalar model parameters (always)
            summary.add_scalar('rbf/alpha', softplus(module.radial_basis_functions._alpha), step)

            # write optional summaries for model parameters
            if args.write_parameter_summaries:
                for name, param in module.named_parameters():
                    splitted_name = name.split('.', 1)
                    if len(splitted_name) > 1:
                        first, last = splitted_name
                    else:
                        first = 'nn'
                        last = splitted_name[0]
                    if param.numel() > 1 and param.requires_grad:  # only tensors get written as histogram
                        summary.add_histogram(first + '/' + last, param.clone().cpu().data.numpy(), step)

            # print progress to consoles
            progress_string = str(step).zfill(len(str(args.max_steps))) + "/" + str(args.max_steps)
            progress_string += " epoch: %6d" % epoch

            for key in loss_weights.keys():
                if loss_weights[key] > 0:
                    progress_string += "\n  " + key + ":\n"
                    progress_string += "    train: %10.8f" % train_errors[key + '_mae']
                    progress_string += "    valid: %10.8f" % valid_errors[key + '_mae']
                    progress_string += "     best: %10.8f" % best_errors[key + '_mae']

            logging.info(progress_string)
            end_time = time.time()
            logging.info("time elapsed: %d", end_time - start_time)
            start_time = end_time

            # reset train metrics
            if args.use_gradient_clipping:
                gradient_norm = 0

            train_errors = empty_error_dict(loss_weights)  # reset train error metrics
            train_batch_num = -1

        # increment step counter
        step += 1

        # save checkpoint (always the last step)
        if step % args.checkpoint_interval == 0:

            # move the latest checkpoint (so it is not overwritten)
            if os.path.isfile(os.path.join(checkpoint_dir, 'latest_checkpoint.pth')):
                os.rename(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'),
                          os.path.join(checkpoint_dir, 'checkpoint_' + str(latest_checkpoint).zfill(10) + '.pth'))

            latest_checkpoint = step

            # overwrite latest checkpoint
            torch.save({
                'ID': ID,
                'args': args,
                'step': step,
                'epoch': epoch,
                'best_errors': best_errors,
                'valid_errors': valid_errors,
                'model_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'exponential_moving_average': (exponential_moving_average.ema if args.use_parameter_averaging else None)
            }, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

            summary.add_text('checkpoints', 'saved checkpoint', step)

            # remove oldest checkpoints
            if args.keep_checkpoints >= 0:  # for negative arguments, all checkpoints are kept
                for file in os.listdir(checkpoint_dir):
                    if file.startswith("checkpoint") and file.endswith('.pth'):

                        checkpoint_step = int(file.split('.pth')[0].split('_')[-1])

                        if checkpoint_step < step - args.checkpoint_interval * args.keep_checkpoints:
                            filename = os.path.join(checkpoint_dir, file)

                            if os.path.isfile(filename):
                                os.remove(filename)

        # decide whether to stop the run based on learning rate
        stop_training = True

        for param_group in optimizer.param_groups:
            stop_training = stop_training and (param_group['lr'] < args.stop_at_learning_rate)

        if stop_training:
            logging.info("Learning rate is smaller than " + str(args.stop_at_learning_rate) + "! Training stopped.")
            break

    # close summary writer
    summary.close()
